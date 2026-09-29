"""偵測有沒有 NVIDIA GPU（用 ctranslate2，不必載入 torch），把 "auto" 換成實際要用的裝置。"""

from __future__ import annotations


_cuda_ok = None


def cuda_available() -> bool:
    """
    這台電腦有沒有可用的 NVIDIA GPU。用 faster-whisper 內建的 ctranslate2 偵測，不必載入 torch；
    偵測本身失敗就當作有（維持原本一律用 GPU 的行為）。
    """
    global _cuda_ok
    if _cuda_ok is None:
        try:
            import ctranslate2
            _cuda_ok = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            _cuda_ok = True
    return _cuda_ok


def resolve_device(setting: str) -> str:
    """"auto" → 有 GPU 就 "cuda"，沒有就 "cpu"；其他值照原樣。"""
    if setting == "auto":
        return "cuda" if cuda_available() else "cpu"
    return setting
