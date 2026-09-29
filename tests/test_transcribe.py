"""裝置自動偵測、Whisper 模型載入、幻覺片語過濾、hotwords 字串。"""
import sys
import types

import pytest

from whisper_app import config, devices, transcribe


def fake_ctranslate2(monkeypatch, count):
    module = types.ModuleType("ctranslate2")
    module.get_cuda_device_count = lambda: count
    monkeypatch.setitem(sys.modules, "ctranslate2", module)


# ---------------- 裝置 ----------------
def test_cuda_available_follows_ctranslate2(monkeypatch):
    fake_ctranslate2(monkeypatch, 1)
    assert devices.cuda_available() is True
    monkeypatch.setattr(devices, "_cuda_ok", None)
    fake_ctranslate2(monkeypatch, 0)
    assert devices.cuda_available() is False


def test_detection_failure_assumes_gpu_like_before(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2", None)      # import 會失敗
    assert devices.cuda_available() is True


def test_result_is_cached(monkeypatch):
    fake_ctranslate2(monkeypatch, 0)
    assert devices.cuda_available() is False
    fake_ctranslate2(monkeypatch, 5)
    assert devices.cuda_available() is False


@pytest.mark.parametrize("setting, gpus, expected", [("auto", 0, "cpu"), ("auto", 1, "cuda"), ("cuda", 0, "cuda"), ("cpu", 1, "cpu")])
def test_resolve_device(monkeypatch, setting, gpus, expected):
    fake_ctranslate2(monkeypatch, gpus)
    assert devices.resolve_device(setting) == expected


# ---------------- Whisper 模型 ----------------
class RecordingWhisper:
    calls = []

    def __init__(self, name, device, compute_type):
        RecordingWhisper.calls.append((name, device, compute_type))


@pytest.mark.parametrize("setting, gpus, expected, warns", [
    ("auto", 0, ("cpu", "int8"), True),
    ("auto", 1, ("cuda", "float16"), False),
    ("cpu", 1, ("cpu", "int8"), False),          # 明確指定 cpu：計算精度自動換成 CPU 支援的
    ("cuda", 0, ("cuda", "float16"), False),      # 明確指定 cuda：照辦，不替使用者改
])
def test_get_model_device_and_compute_type(monkeypatch, setting, gpus, expected, warns):
    fake_ctranslate2(monkeypatch, gpus)
    RecordingWhisper.calls.clear()
    monkeypatch.setattr(transcribe, "WhisperModel", RecordingWhisper)
    monkeypatch.setattr(config, "WHISPER_DEVICE", setting)
    logs = []
    transcribe.get_model(logs.append)
    assert RecordingWhisper.calls == [("large-v3",) + expected]
    assert any("改用 CPU" in x for x in logs) is warns


def test_get_model_loads_only_once(monkeypatch):
    fake_ctranslate2(monkeypatch, 1)
    RecordingWhisper.calls.clear()
    monkeypatch.setattr(transcribe, "WhisperModel", RecordingWhisper)
    first = transcribe.get_model(lambda m: None)
    assert transcribe.get_model(lambda m: None) is first and len(RecordingWhisper.calls) == 1


# ---------------- 幻覺片語 ----------------
PHRASE = "ご視聴ありがとうございました"


@pytest.mark.parametrize("text, no_speech, logprob, expected", [
    (PHRASE + "。", 0.6, -0.3, True),                       # 整句是片語 + 無語音機率高
    (PHRASE, 0.02, -0.3, False),                            # 真的有人說：不刪
    (PHRASE, 0.02, -1.4, True),                             # 片語但平均信心很低
    ("それでは" + PHRASE + "皆さん", 0.9, -2.0, False),      # 只是長句的一部分
    ("ご視聴 ありがとう ございました！", 0.9, None, True),    # 空白、全形標點不影響
    ("ありがとうございました", 0.9, -2.0, False),             # 太一般的話不在清單裡
    (PHRASE, None, None, False),                            # 沒有機率資訊：不刪
    ("今日は天気がいいですね", 0.9, -2.0, False),
    ("チャンネル登録をお願いします。", 0.7, -0.4, True),
])
def test_looks_like_hallucination(text, no_speech, logprob, expected):
    assert transcribe.looks_like_hallucination(text, no_speech, logprob) is expected


def test_hallucination_filter_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(config, "FILTER_HALLUCINATIONS", False)
    assert transcribe.looks_like_hallucination(PHRASE, 0.9, -2.0) is False


def test_phrase_key_normalizes_width_and_punctuation():
    assert transcribe._phrase_key("ＡＢＣ　あ、い。 ①") == "abcあい1"


# ---------------- hotwords ----------------
def test_build_hotwords_with_japanese_punctuation(monkeypatch):
    monkeypatch.setattr(config, "HOTWORDS_JA_PUNCTUATION", True)
    assert transcribe.build_hotwords(["クイズ", "東大"]) == "クイズ、東大。"
    assert transcribe.build_hotwords([]) == ""


def test_build_hotwords_plain_style(monkeypatch):
    monkeypatch.setattr(config, "HOTWORDS_JA_PUNCTUATION", False)
    assert transcribe.build_hotwords(["クイズ", "東大"]) == "クイズ, 東大"
