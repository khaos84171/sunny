"""測試用的替身：不需要真的裝 faster-whisper（也不需要 GPU）。個別測試會用 monkeypatch 換成自己的假模型。"""


class WhisperModel:
    def __init__(self, model_size_or_path, device="cpu", compute_type="default", **kwargs):
        self.args = (model_size_or_path, device, compute_type)


def decode_audio(*args, **kwargs):
    raise NotImplementedError("測試要自己換掉 aligner.decode_audio")
