"""時間戳對齊的主程式這一側:管理常駐的 align_worker.py 程序，把工作交給它。"""

from __future__ import annotations

import collections
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from faster_whisper import decode_audio

from . import config
from .jobs import JOBS, _NO_WINDOW, _child_env, _console_python


class AlignServer:
    """
    對齊程序（align_worker.py --serve）的管理。載入 wav2vec2 要好幾秒，連續處理多個檔案時
    程序留著給下一個檔案用；閒置超過 ALIGN_KEEP_ALIVE_SEC 秒它會自己結束並釋放記憶體。
    取消、關閉視窗時 JOBS 會把它一起停掉，下次要用再重新啟動。
    """

    def __init__(self):
        self._proc = None

    def _ensure(self):
        """回傳可用的對齊程序；還沒有或已經結束了就重新啟動。"""
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        JOBS.release(self._proc)  # 結束了的，從登記處拿掉
        self._proc = JOBS.spawn(
            [_console_python(), str(config.ALIGN_WORKER), "--serve", str(config.ALIGN_KEEP_ALIVE_SEC)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=_child_env(HF_HUB_DISABLE_SYMLINKS_WARNING="1"), creationflags=_NO_WINDOW,
        )
        return self._proc

    def _drop(self, proc) -> None:
        """丟掉一個確定不能用的對齊程序（等它真的結束），下次 _ensure 一定開新的。
        剛結束的程序有一瞬間 poll() 還是 None，重試時不能再拿到它。"""
        JOBS.release(proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self._proc = None

    def run(self, job_json: Path, result_json: Path, log_func, progress_func):
        """交一個工作給對齊程序並等它做完。回傳 (代碼, 最後幾行輸出)；代碼 0 = 成功。"""
        request = json.dumps({"job": str(job_json), "result": str(result_json)}, ensure_ascii=False) + "\n"
        for attempt in (1, 2):
            if self._proc is None or self._proc.poll() is not None:
                log_func("[對齊] 啟動對齊程序…")
            proc = self._ensure()
            try:
                proc.stdin.write(request)
                proc.stdin.flush()
            except OSError:  # 程序剛好在閒置逾時的瞬間結束了
                self._drop(proc)
                if attempt == 1:
                    continue
                raise
            started, code = False, None
            tail = collections.deque(maxlen=15)
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                if line.startswith("START"):
                    started = True
                elif line.startswith("END "):
                    code = int(line.split()[1])
                    break
                elif line.startswith("PROGRESS "):
                    done, total = line.split()[1:3]
                    progress_func(int(done) / max(int(total), 1))
                elif line.startswith("LOG "):
                    log_func("[對齊] " + line[4:])
                elif line.startswith("ERROR "):
                    log_func("!!! [對齊] " + line[6:])
                    tail.append(line[6:])
                elif line.strip():
                    print(line)  # transformers 的警告、下載進度等只寫進 log 檔
                    tail.append(line)
            JOBS.check()  # 被取消時對齊程序是被停掉的，輸出中斷，要先判斷這個
            if code is None:  # 對齊程序在做完之前就沒了
                if not started and attempt == 1:
                    self._drop(proc)
                    continue  # 還沒開始做就結束了（剛好閒置逾時）：重開一個再試一次
                code = proc.wait() or 1
            return code, tail


ALIGN_SERVER = AlignServer()


def run_alignment(audio_path: str, segments: list[dict], log_func, progress_func) -> dict:
    """
    呼叫 align_worker.py 用 wav2vec2 強制對齊。
    segments: [{"start", "end", "text", "words"(可省略，list[Word])}, ...]
    回傳 {"duration", "spans", "confs", "wide", "word_spans", "checks"}，每個 list 都跟 segments 一樣長；
    個別片段對不上時該項是 None，不會整份失敗。
    """
    if not config.ALIGN_WORKER.exists():
        raise FileNotFoundError(f"找不到 {config.ALIGN_WORKER.name}，請把它放在 {config.BASE_DIR}")

    log_func(f"[對齊] 讀取音訊：{audio_path}")
    audio = decode_audio(audio_path, sampling_rate=config.SAMPLE_RATE)
    duration = len(audio) / config.SAMPLE_RATE

    with tempfile.TemporaryDirectory(prefix="whisper_align_") as tmp:
        tmp = Path(tmp)
        audio_npy = tmp / "audio.npy"
        job_json = tmp / "job.json"
        result_json = tmp / "result.json"

        np.save(audio_npy, audio.astype(np.float32, copy=False))
        del audio
        job_json.write_text(json.dumps({
            "audio_npy": str(audio_npy),
            "segments": [
                {
                    "start": s["start"], "end": s["end"], "text": s["text"],
                    # 有詞的切法就一起送，worker 會傳回每個詞對齊後的時間（給拆分用）
                    "words": [w.word for w in s["words"]] if s.get("words") else None,
                }
                for s in segments
            ],
            "model_name": config.ALIGN_MODEL_NAME,
            "device": config.ALIGN_DEVICE,
            "sample_rate": config.SAMPLE_RATE,
            "params": {
                "pad_sec": config.ALIGN_PAD_SEC,
                "edge_sec": config.ALIGN_EDGE_SEC,
                "wide_sec": config.ALIGN_WIDE_SEC,
                "batch_sec": config.ALIGN_BATCH_SEC,
                "low_conf": config.ALIGN_LOW_CONF,
                "max_inner_gap": config.ALIGN_MAX_INNER_GAP,
                # 舊版 align_worker.py 不認得這一項就會略過（等於只用 CTC）
                "refine": {
                    "enabled": bool(config.ALIGN_REFINE),
                    "back_sec": config.ALIGN_REFINE_BACK_SEC,
                    "fwd_sec": config.ALIGN_REFINE_FWD_SEC,
                    "end_fwd_sec": config.ALIGN_REFINE_END_FWD_SEC,
                    "end_back_sec": config.ALIGN_REFINE_END_BACK_SEC,
                    "min_contrast_db": config.ALIGN_REFINE_MIN_CONTRAST_DB,
                    "thr_frac": config.ALIGN_REFINE_THRESHOLD,
                    "agree_sec": config.ALIGN_AGREE_SEC,
                },
            },
        }, ensure_ascii=False), encoding="utf-8")

        returncode, tail = ALIGN_SERVER.run(job_json, result_json, log_func, progress_func)
        if returncode != 0 or not result_json.exists():
            detail = "\n".join(tail) if tail else "（沒有輸出）"
            if "--serve" in detail:  # 舊版 align_worker.py 不認得常駐模式
                detail += "\n（align_worker.py 是舊版，請把它跟 w1_1.py 一起更新）"
            raise RuntimeError(f"對齊程序失敗（代碼 {returncode}）：\n{detail}")

        result = json.loads(result_json.read_text(encoding="utf-8"))

    n = len(segments)
    spans = result["spans"]
    if len(spans) != n:
        raise RuntimeError(f"對齊結果數量不符（送出 {n} 條，收到 {len(spans)} 條）")
    return {
        "duration": duration,
        "spans": spans,
        "confs": result.get("confs") or [None] * n,
        "wide": result.get("wide") or [False] * n,
        # 舊版 align_worker.py 沒有這一項：拆分會退回用 Whisper 的詞時間
        "word_spans": result.get("word_spans") or [None] * n,
        # 每個詞的聲音交叉檢查 [起點狀態, 起點修正秒數, 終點狀態, 終點修正秒數]；舊版 align_worker.py 沒有
        "checks": result.get("checks") or [None] * n,
    }
