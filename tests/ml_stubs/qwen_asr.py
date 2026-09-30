"""測試用的假 qwen_asr（說明見 fake_asr_common.py）。"""
from types import SimpleNamespace

from fake_asr_common import Module, answer, check_installed, log

check_installed("qwen", "qwen_asr")


class Qwen3ASRModel:
    @classmethod
    def from_pretrained(cls, name, dtype=None, device_map=None, **kwargs):
        log(f"load qwen {name} {device_map} {dtype}")
        return cls()

    def __init__(self):
        self.model = Module("qwen")

    def transcribe(self, audio, context="", language=None, return_time_stamps=False):
        log(f"qwen transcribe n={len(audio)} context={context!r} language={language!r}")
        return [SimpleNamespace(language="Japanese", text=answer("qwen", wav)) for wav, sr in audio]
