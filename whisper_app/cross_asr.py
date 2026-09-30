"""交叉比對的主程式這一側:把每條字幕的那段聲音交給 asr_worker.py（Qwen3-ASR、Parakeet）再聽一次，收回辨識結果。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from faster_whisper import decode_audio

from . import config
from .jobs import JOBS, JobCancelled, _console_python
from .workers import WorkerServer


SERVERS: dict[str, WorkerServer] = {}   # 每種模型一個常駐程序（可能各自用不同的 Python 環境）


def backend_label(name: str) -> str:
    return (config.CROSS_BACKENDS.get(name) or {}).get("label", name)


def _server(name: str) -> WorkerServer:
    if name not in SERVERS:
        def command() -> list[str]:
            python = (config.CROSS_BACKENDS[name].get("python") or "").strip() or _console_python()
            return [python, str(config.ASR_WORKER), "--serve", str(config.CROSS_KEEP_ALIVE_SEC)]
        label = backend_label(name)
        SERVERS[name] = WorkerServer(f"交叉比對:{label}", f" {label} 程序", command)
    return SERVERS[name]


def clip_spans(segments: list[dict], duration: float) -> list[list[float]]:
    """每條字幕要送去辨識的聲音範圍：前後各多取 CROSS_PAD_SEC 秒，但不伸進前後句的時間裡。"""
    spans = []
    pad = config.CROSS_PAD_SEC
    for i, seg in enumerate(segments):
        lo, hi = seg["start"] - pad, seg["end"] + pad
        if i > 0:
            lo = max(lo, min(segments[i - 1]["end"], seg["start"]))
        if i + 1 < len(segments):
            hi = min(hi, max(segments[i + 1]["start"], seg["end"]))
        spans.append([round(max(0.0, lo), 3), round(min(duration, hi), 3)])
    return spans


def run_cross_asr(audio_path: str, segments: list[dict], hotwords: list[str],
                  log_func, progress_func) -> tuple[list[dict], list[str]]:
    """
    用 CROSS_MODELS 裡的每個模型辨識每條字幕的聲音。
    回傳 (每條字幕 {模型名稱: 結果 或 None}, 成功跑完的模型名稱)。某個模型失敗（沒裝、顯存不足…）只記在日誌、
    那個模型的結果全部是 None，不會讓整個檔案失敗；取消則照常往外丟。
    """
    if not config.ASR_WORKER.exists():
        raise FileNotFoundError(f"找不到 {config.ASR_WORKER.name}，請把它放在 {config.BASE_DIR}")
    names = []
    for name in config.CROSS_MODELS:
        if name in config.CROSS_BACKENDS:
            names.append(name)
        else:
            log_func(f"!!! [交叉比對] 不認得的模型「{name}」（whisper_app/config.py 的 CROSS_MODELS），略過")
    hyps = [{name: None for name in names} for _ in segments]
    if not names or not segments:
        return hyps, []

    audio = JOBS.run_interruptibly(decode_audio, audio_path, sampling_rate=config.SAMPLE_RATE)  # 長影片要解碼好幾秒
    duration = len(audio) / config.SAMPLE_RATE
    spans = clip_spans(segments, duration)
    used = []
    with tempfile.TemporaryDirectory(prefix="whisper_cross_") as tmp:
        tmp = Path(tmp)
        audio_npy = tmp / "audio.npy"
        np.save(audio_npy, audio.astype(np.float32, copy=False))
        del audio
        for k, name in enumerate(names):
            cfg = config.CROSS_BACKENDS[name]
            label = backend_label(name)
            job_json, result_json = tmp / f"job_{name}.json", tmp / f"result_{name}.json"
            options = {"batch_size": cfg.get("batch_size", 8)}
            if cfg.get("language"):
                options["language"] = cfg["language"]
            if cfg.get("use_hotwords") and hotwords:
                options["context"] = "、".join(hotwords)
            if "decoder" in cfg:
                options["decoder"] = cfg["decoder"]
            job_json.write_text(json.dumps({
                "backend": name, "model_name": cfg["model"], "device": config.CROSS_DEVICE,
                "sample_rate": config.SAMPLE_RATE, "audio_npy": str(audio_npy), "clips": spans, "options": options,
            }, ensure_ascii=False), encoding="utf-8")
            log_func(f"[交叉比對] 用 {label} 再聽一次（{len(spans)} 條）…")
            try:
                code, tail = _server(name).run(
                    job_json, result_json, log_func,
                    lambda frac, k=k: progress_func((k + frac) / len(names)))
                if code != 0 or not result_json.exists():
                    raise RuntimeError("\n".join(tail) if tail else f"代碼 {code}，沒有輸出")
                texts = json.loads(result_json.read_text(encoding="utf-8"))["texts"]
                if len(texts) != len(segments):
                    raise RuntimeError(f"結果數量不符（送出 {len(segments)} 條，收到 {len(texts)} 條）")
            except JobCancelled:
                raise
            except Exception as e:
                log_func(f"!!! [交叉比對] {label} 失敗，這次不用它：{e}")
                continue
            for h, text in zip(hyps, texts):
                h[name] = text
            used.append(name)
            progress_func((k + 1) / len(names))
    return hyps, used
