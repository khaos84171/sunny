"""時間戳對齊的主程式這一側:管理常駐的 align_worker.py 程序，把工作交給它。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from faster_whisper import decode_audio

from . import config
from .jobs import JOBS, _console_python
from .srt_io import format_timestamp
from .timing import detect_timeline_offset
from .workers import WorkerServer


def _align_command() -> list[str]:
    return [_console_python(), str(config.ALIGN_WORKER), "--serve", str(config.ALIGN_KEEP_ALIVE_SEC)]


# 對齊程序（align_worker.py --serve）。載入 wav2vec2 要好幾秒，連續處理多個檔案時程序留著給下一個檔案用；
# 閒置超過 ALIGN_KEEP_ALIVE_SEC 秒它會自己結束並釋放記憶體。
ALIGN_SERVER = WorkerServer("對齊", "對齊程序", _align_command)


def _shift_result(result: dict, offset: float) -> dict:
    """把對齊結果的所有時間加上 offset 秒（把「影片的時間軸」換回「字幕原本的時間軸」）。"""
    def shift_pairs(pairs):
        return None if pairs is None else [None if p is None else [p[0] + offset, p[1] + offset] for p in pairs]
    result["spans"] = [None if s is None else [s[0] + offset, s[1] + offset] for s in result["spans"]]
    result["word_spans"] = [shift_pairs(w) for w in result["word_spans"]]
    return result


def run_alignment(audio_path: str, segments: list[dict], log_func, progress_func, detect_offset: bool = False) -> dict:
    """
    呼叫 align_worker.py 用 wav2vec2 強制對齊。
    segments: [{"start", "end", "text", "words"(可省略，list[Word])}, ...]
    回傳 {"duration", "offset", "spans", "confs", "wide", "word_spans", "checks"}，每個 list 都跟 segments 一樣長；
    個別片段對不上時該項是 None，不會整份失敗。
    detect_offset=True（只對齊現有字幕時用）：字幕的時間軸整個差了整數小時（剪輯軟體匯出的字幕常從 01:00:00 開始）
    就先扣掉再對齊，回傳的時間（含 duration）換回字幕原本的時間軸，offset 是扣掉的秒數（沒有偏移 = 0.0）。
    字幕跟影片明顯對不起來時丟 ValueError。
    """
    if not config.ALIGN_WORKER.exists():
        raise FileNotFoundError(f"找不到 {config.ALIGN_WORKER.name}，請把它放在 {config.BASE_DIR}")

    log_func(f"[對齊] 讀取音訊：{audio_path}")
    audio = JOBS.run_interruptibly(decode_audio, audio_path, sampling_rate=config.SAMPLE_RATE)  # 長影片要解碼好幾秒
    duration = len(audio) / config.SAMPLE_RATE

    offset = detect_timeline_offset(segments, duration) if detect_offset else 0.0
    if offset:
        log_func(f"[對齊] 字幕的時間軸從 {format_timestamp(offset)} 開始（剪輯軟體常見的時間軸起點），"
                 f"先整體提早 {offset / 3600:g} 小時再對齊；輸出的字幕維持原本的時間軸")
    elif detect_offset:
        late = sum(1 for s in segments if s["start"] >= duration)
        if late:
            log_func(f"[對齊] 注意：有 {late} 條字幕的起點超過影片長度（{format_timestamp(duration)}），這些沒辦法對齊")

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
                    "start": s["start"] - offset, "end": s["end"] - offset, "text": s["text"],
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

        raw = json.loads(result_json.read_text(encoding="utf-8"))

    n = len(segments)
    if len(raw["spans"]) != n:
        raise RuntimeError(f"對齊結果數量不符（送出 {n} 條，收到 {len(raw['spans'])} 條）")
    result = {
        "duration": duration,
        "offset": offset,
        "spans": raw["spans"],
        "confs": raw.get("confs") or [None] * n,
        "wide": raw.get("wide") or [False] * n,
        # 舊版 align_worker.py 沒有這一項：拆分會退回用 Whisper 的詞時間
        "word_spans": raw.get("word_spans") or [None] * n,
        # 每個詞的聲音交叉檢查 [起點狀態, 起點修正秒數, 終點狀態, 終點修正秒數]；舊版 align_worker.py 沒有
        "checks": raw.get("checks") or [None] * n,
    }
    if offset:
        _shift_result(result, offset)
        result["duration"] += offset
    return result
