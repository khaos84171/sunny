"""測試用的假 nemo.collections.asr（說明見 ml_stubs/fake_asr_common.py）。"""
from types import SimpleNamespace

from fake_asr_common import Module, answer, check_installed, log

check_installed("parakeet", "nemo")


class ASRModel(Module):
    @classmethod
    def from_pretrained(cls, model_name, map_location=None):
        log(f"load parakeet {model_name} {map_location}")
        return cls("parakeet")

    def eval(self):
        return self

    def change_decoding_strategy(self, decoder_type=None, **kwargs):
        log(f"parakeet decoder {decoder_type}")

    def transcribe(self, audio, batch_size=4, verbose=True, **kwargs):
        log(f"parakeet transcribe n={len(audio)}")
        return [SimpleNamespace(text=answer("parakeet", a)) for a in audio]


models = SimpleNamespace(ASRModel=ASRModel)
