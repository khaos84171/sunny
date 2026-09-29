"""Whisper 轉錄相關:載入模型、hotwords、過濾 Whisper 在靜音上編出來的固定片語。"""

from __future__ import annotations

import unicodedata

from faster_whisper import WhisperModel

from . import config
from .devices import resolve_device


# === 模型快取：第一次轉錄時載入，之後一直重用 ===
_loaded_model = None


def get_model(log_func) -> WhisperModel:
    global _loaded_model
    if _loaded_model is None:
        device = resolve_device(config.WHISPER_DEVICE)
        compute_type = config.WHISPER_COMPUTE_TYPE if device.startswith("cuda") else config.WHISPER_CPU_COMPUTE_TYPE
        if device == "cpu" and config.WHISPER_DEVICE == "auto":
            log_func(f"偵測不到可用的 NVIDIA GPU，Whisper 改用 CPU（{compute_type}），速度會慢很多")
        log_func(f"載入模型中：{config.WHISPER_MODEL}（{device}／{compute_type}）…（只有第一次會比較久）")
        _loaded_model = WhisperModel(config.WHISPER_MODEL, device=device, compute_type=compute_type)
        log_func("模型載入完成。")
    return _loaded_model


def _phrase_key(text: str) -> str:
    """比對片語用：全形半形統一、去掉標點與空白。"""
    return "".join(ch for ch in unicodedata.normalize("NFKC", text).lower() if ch.isalnum())


_HALLUCINATION_KEYS = frozenset(_phrase_key(p) for p in config.HALLUCINATION_PHRASES)


def looks_like_hallucination(text: str, no_speech_prob, avg_logprob) -> bool:
    """整條字幕只有已知的幻覺片語，而且 Whisper 自己也覺得那裡不像有人說話。"""
    if not config.FILTER_HALLUCINATIONS or _phrase_key(text) not in _HALLUCINATION_KEYS:
        return False
    return ((no_speech_prob is not None and no_speech_prob >= config.HALLUCINATION_MIN_NO_SPEECH_PROB)
            or (avg_logprob is not None and avg_logprob <= config.HALLUCINATION_MAX_AVG_LOGPROB))


def build_hotwords(words: list[str]) -> str:
    """把勾選的 hotwords 串成傳給 Whisper 的字串。"""
    if not words:
        return ""
    if config.HOTWORDS_JA_PUNCTUATION:
        return "、".join(words) + "。"
    return ", ".join(words)
