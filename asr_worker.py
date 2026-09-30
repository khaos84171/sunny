"""
交叉比對用的語音辨識工作程序（由 w1_1.py 自動呼叫，不需要手動執行，請跟 w1_1.py 放在同一個資料夾）。

Whisper 轉錄完後，主程式把每一條字幕的那一段聲音交給這裡，用另一個獨立的辨識模型再聽一次，
結果拿回去跟 Whisper 投票（多數決修正聽錯的字，見 whisper_app/crosscheck.py）。支援兩種模型：
    qwen3      Qwen3-ASR（阿里巴巴 Qwen 團隊，pip install -U qwen-asr）
               可以給「提示」（context）：主程式會把勾選的 hotwords 傳進來，人名比較不會聽錯。
    parakeet   NVIDIA Parakeet TDT-CTC 日文版（pip install -U "nemo_toolkit[asr]"）

跟 align_worker.py 一樣用獨立程序跑（PyTorch 與 faster-whisper 的 cuDNN 版本衝突，說明見 align_worker.py 開頭），
而且兩種模型各自一個程序：它們需要的 transformers／NeMo 版本不同，可以分別裝在不同的虛擬環境
（whisper_app/config.py 的 CROSS_BACKENDS[...]["python"]）。

用法：
    python asr_worker.py --serve <閒置秒數>
        常駐模式（w1_1.py 用這個）：通訊協定跟 align_worker.py 完全一樣——工作一行一個從 stdin 讀進來
        （JSON：{"job": job.json 路徑, "result": result.json 路徑}），開始做印 "START"，做完印 "END <0=成功>"。
        模型載入一次就重複用；每個工作做完先把模型搬到 CPU 釋放顯存。閒置超過指定秒數自己結束；0 = 做完一個就結束。
    python asr_worker.py job.json result.json
        單次模式（手動測試用）。
    job.json    {"backend": "qwen3" 或 "parakeet", "model_name", "device", "sample_rate", "audio_npy",
                 "clips": [[起點秒, 終點秒], ...], "options": {"batch_size", "language", "context", "decoder"}}
    result.json {"texts": [辨識結果（聽不到任何字是 ""）, ...]}，跟 clips 一樣長
stdout 每行一則訊息給主程式："LOG <文字>"、"PROGRESS <已完成> <總數>"、"ERROR <文字>"、"START"、"END <代碼>"
"""

import json
import queue
import sys
import threading
import traceback

import numpy as np

MIN_CLIP_SEC = 0.3   # 太短的片段前後補靜音到這個長度（模型對極短的輸入常常什麼都不輸出）


def emit(kind: str, msg) -> None:
    print(f"{kind} {msg}", flush=True)


def _resolve_device(device: str) -> str:
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class _Backend:
    """一種辨識模型：載入、辨識一批聲音、搬到 CPU 讓出顯存／搬回來。"""

    label = ""

    def __init__(self, model_name: str, device: str):
        self.requested = (type(self).__name__, model_name, device)
        self.device = _resolve_device(device)
        if device == "auto" and self.device == "cpu":
            emit("LOG", "偵測不到可用的 NVIDIA GPU，改用 CPU（會慢很多）")
        emit("LOG", f"載入 {self.label} 模型：{model_name}（{self.device}）…（第一次使用會自動下載）")
        self._load(model_name)
        self._parked = False
        emit("LOG", f"{self.label} 模型載入完成")

    def _modules(self) -> list:
        return []

    def park(self) -> None:
        """做完一個工作：模型搬到 CPU，顯存留給 Whisper／Demucs／對齊。"""
        if not self.device.startswith("cuda"):
            return
        import torch
        for module in self._modules():
            module.to("cpu")
        torch.cuda.empty_cache()
        self._parked = True

    def unpark(self) -> None:
        if self._parked:
            for module in self._modules():
                module.to(self.device)
            self._parked = False
        emit("LOG", f"沿用已載入的 {self.label} 模型")


class Qwen3Backend(_Backend):
    label = "Qwen3-ASR"

    def _load(self, model_name: str) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = Qwen3ASRModel.from_pretrained(model_name, dtype=dtype, device_map=self.device,
                                                   max_inference_batch_size=-1, max_new_tokens=512)

    def _modules(self) -> list:
        return [self.model.model]

    def transcribe(self, clips: list, sr: int, opts: dict) -> list[str]:
        results = self.model.transcribe(audio=[(c, sr) for c in clips], context=opts.get("context") or "",
                                        language=opts.get("language") or None)
        return [r.text for r in results]


class ParakeetBackend(_Backend):
    label = "Parakeet"

    def _load(self, model_name: str) -> None:
        import torch
        import nemo.collections.asr as nemo_asr
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location=torch.device(self.device))
        self.model.eval()
        self._decoder = "tdt"  # 模型本來的解碼器

    def _modules(self) -> list:
        return [self.model]

    def transcribe(self, clips: list, sr: int, opts: dict) -> list[str]:
        decoder = opts.get("decoder") or "tdt"
        if decoder != self._decoder and hasattr(self.model, "change_decoding_strategy"):
            try:  # 混合 TDT-CTC 模型可以切換解碼方式；"tdt" 是它本來的解碼器（在 NeMo 裡叫 "rnnt"）
                self.model.change_decoding_strategy(decoder_type="ctc" if decoder == "ctc" else "rnnt")
            except Exception as e:
                emit("LOG", f"無法切換成 {decoder} 解碼（{e}），用模型預設的")
            self._decoder = decoder
        out = self.model.transcribe(list(clips), batch_size=len(clips), verbose=False)
        if isinstance(out, tuple):  # 舊版 NeMo：(最佳結果, 所有候選)
            out = out[0]
        return [o if isinstance(o, str) else getattr(o, "text", "") for o in out]


BACKENDS = {"qwen3": Qwen3Backend, "parakeet": ParakeetBackend}


def cut_clips(audio: np.ndarray, sr: int, spans: list) -> list:
    """依 [起點秒, 終點秒] 切出每一段聲音；太短的前後補靜音。"""
    clips = []
    min_len = int(MIN_CLIP_SEC * sr)
    for s, e in spans:
        a = max(0, int(round(s * sr)))
        b = min(len(audio), max(a, int(round(e * sr))))
        clip = np.asarray(audio[a:b], dtype=np.float32)
        if len(clip) < min_len:
            pad = min_len - len(clip)
            clip = np.pad(clip, (pad // 2, pad - pad // 2))
        clips.append(clip)
    return clips


def asr_job(backend: _Backend, job: dict) -> dict:
    sr = int(job.get("sample_rate", 16000))
    audio = np.load(job["audio_npy"], mmap_mode="r")
    opts = job.get("options") or {}
    clips = cut_clips(audio, sr, job["clips"])
    batch = max(1, int(opts.get("batch_size") or 8))
    texts: list = []
    emit("PROGRESS", f"0 {len(clips)}")
    for i in range(0, len(clips), batch):
        chunk = clips[i:i + batch]
        out = backend.transcribe(chunk, sr, opts)
        if len(out) != len(chunk):
            raise RuntimeError(f"辨識結果數量不符（送出 {len(chunk)} 段，收到 {len(out)} 段）")
        texts.extend((t or "").strip() for t in out)
        emit("PROGRESS", f"{len(texts)} {len(clips)}")
    return {"texts": texts}


def _make_backend(job: dict) -> _Backend:
    name = job.get("backend")
    if name not in BACKENDS:
        raise ValueError(f"不認得的模型種類：{name!r}（可用的：{', '.join(BACKENDS)}）")
    return BACKENDS[name](job["model_name"], job.get("device", "auto"))


def run_once(job_path: str, result_path: str) -> None:
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    result = asr_job(_make_backend(job), job)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


_INSTALL_HINT = {"qwen_asr": "qwen-asr", "nemo": "nemo_toolkit[asr]"}


def _missing_package_message(e: ImportError) -> str:
    missing = (getattr(e, "name", None) or "").split(".")[0] or "qwen-asr / nemo_toolkit"
    return f"缺少套件 {missing}（請執行 pip install -U \"{_INSTALL_HINT.get(missing, missing)}\"）"


def serve(idle_sec: float) -> None:
    """常駐模式：從 stdin 一行一個工作，模型只載入一次；idle_sec <= 0 表示做完一個就結束。"""
    lines: queue.Queue = queue.Queue()

    def read_stdin() -> None:
        for line in sys.stdin:
            lines.put(line)
        lines.put(None)  # stdin 被關掉了：主程式不需要我了

    threading.Thread(target=read_stdin, daemon=True).start()
    backend = None
    while True:
        try:
            line = lines.get(timeout=idle_sec) if idle_sec > 0 else lines.get()
        except queue.Empty:
            if backend is not None:
                emit("LOG", f"閒置超過 {idle_sec:g} 秒，{backend.label} 程序結束（下次會重新載入模型）")
            return
        if line is None:
            return
        if not line.strip():
            continue
        request = json.loads(line)
        emit("START", "")
        try:
            with open(request["job"], encoding="utf-8") as f:
                job = json.load(f)
            cls = BACKENDS.get(job.get("backend"))
            key = (cls.__name__ if cls else None, job["model_name"], job.get("device", "auto"))
            if backend is not None and backend.requested == key:
                backend.unpark()
            else:
                backend = None  # 先放掉舊的再載入新的
                backend = _make_backend(job)
            result = asr_job(backend, job)
            with open(request["result"], "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
            backend.park()
        except ImportError as e:
            traceback.print_exc()
            sys.stdout.flush()
            emit("ERROR", _missing_package_message(e))
            emit("END", 2)
            sys.exit(2)
        except Exception as e:
            traceback.print_exc()
            sys.stdout.flush()
            emit("ERROR", f"{type(e).__name__}: {e}")
            emit("END", 1)
            sys.exit(1)  # 出錯之後模型狀態不可信（例如顯存不足），結束讓主程式下次重開
        emit("END", 0)
        if idle_sec <= 0:
            return


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--serve":
        serve(float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
    else:
        run_once(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        emit("ERROR", _missing_package_message(e))
        sys.exit(2)
    except Exception as e:
        traceback.print_exc()
        sys.stdout.flush()
        emit("ERROR", f"{type(e).__name__}: {e}")
        sys.exit(1)
