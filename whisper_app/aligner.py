"""時間戳對齊的主程式這一側:管理常駐的 align_worker.py 程序，把工作交給它。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from faster_whisper import decode_audio

from . import config
from .jobs import _console_python
from .workers import WorkerServer


def _align_command() -> list[str]:
    return [_console_python(), str(config.ALIGN_WORKER), "--serve", str(config.ALIGN_KEEP_ALIVE_SEC)]


# 對齊程序（align_worker.py --serve）。載入 wav2vec2 要好幾秒，連續處理多個檔案時程序留著給下一個檔案用；
# 閒置超過 ALIGN_KEEP_ALIVE_SEC 秒它會自己結束並釋放記憶體。
ALIGN_SERVER = WorkerServer("對齊", "對齊程序", _align_command)


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
