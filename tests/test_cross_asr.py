"""交叉比對的模型呼叫：透過真的 asr_worker.py 子程序（配假的 qwen_asr / nemo）跑，涵蓋結果回收、提示詞、
程序重用、顯存搬移、某個模型沒裝時不影響其他的；以及 process_file 裡的交叉比對（修正、檔名、進度、失敗時照常輸出）。"""
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from helpers import TESTS
from whisper_app import config, cross_asr, pipeline, runtime
from whisper_app.jobs import JOBS
from whisper_app.models import Word

SR = 16000
SPANS = [(1.0, 3.0), (4.0, 6.0), (7.0, 9.0)]


def make_audio():
    """每條字幕的位置放一段振幅 = 編號/100 的方波（編號從 1 開始），其他地方靜音：假模型靠振幅認出是第幾條。"""
    audio = np.zeros(SR * 10, dtype=np.float32)
    for k, (s, e) in enumerate(SPANS, start=1):
        n = int((e - s) * SR)
        audio[int(s * SR):int(s * SR) + n] = (k / 100) * np.where(np.arange(n) % 2, 1.0, -1.0)
    return audio


SEGMENTS = [{"start": s, "end": e, "text": t, "words": None} for (s, e), t in zip(SPANS, ["今日は天気", "こんにちは", "ご視聴"])]
SCRIPT = {"qwen": {"1": "今日は電気", "2": "こんにちは"}, "parakeet": {"1": "今日は電気。", "2": "こんにちは"}}


@pytest.fixture
def workers(monkeypatch, tmp_path):
    script = tmp_path / "script.json"
    script.write_text(json.dumps(SCRIPT, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(TESTS / "ml_stubs"), str(TESTS.parent)]))
    monkeypatch.setenv("FAKE_MODEL_LOG", str(tmp_path / "model.log"))
    monkeypatch.setenv("FAKE_ASR_SCRIPT", str(script))
    monkeypatch.setattr(cross_asr, "decode_audio", lambda path, sampling_rate=16000: make_audio())
    monkeypatch.setattr(config, "CROSS_KEEP_ALIVE_SEC", 60)

    class W:
        log = tmp_path / "model.log"

        @staticmethod
        def model_log() -> str:
            return W.log.read_text(encoding="utf-8") if W.log.exists() else ""

        @staticmethod
        def run(segments=SEGMENTS, hotwords=("伊沢拓司",)):
            logs, prog = [], []
            hyps, used = cross_asr.run_cross_asr("/x/a.wav", segments, list(hotwords), logs.append, prog.append)
            return hyps, used, logs, prog
    yield W
    JOBS.cancel()


def test_both_models_hear_every_subtitle(workers):
    hyps, used, logs, prog = workers.run()
    assert used == ["qwen3", "parakeet"]
    assert hyps == [{"qwen3": "今日は電気", "parakeet": "今日は電気。"}, {"qwen3": "こんにちは", "parakeet": "こんにちは"},
                    {"qwen3": "", "parakeet": ""}]                                   # 劇本沒寫 = 聽不到任何字
    assert prog[-1] == 1.0 and prog == sorted(prog)
    assert any("啟動 Qwen3-ASR 程序" in x for x in logs) and any("啟動 Parakeet 程序" in x for x in logs)
    log = workers.model_log()
    assert "load qwen Qwen/Qwen3-ASR-1.7B cpu float32" in log and "load parakeet nvidia/parakeet-tdt_ctc-0.6b-ja cpu" in log
    assert "context='伊沢拓司' language='Japanese'" in log                         # hotwords 當成 Qwen3-ASR 的提示
    assert "parakeet decoder" not in log                                           # 預設 TDT：不用切換解碼器


def test_workers_are_reused_and_park_the_model_on_the_cpu_between_files(workers, monkeypatch):
    monkeypatch.setenv("FAKE_CUDA", "1")
    spawned = []
    real_spawn = JOBS.spawn
    monkeypatch.setattr(JOBS, "spawn", lambda cmd, **kw: (spawned.append(cmd), real_spawn(cmd, **kw))[1])
    workers.run()
    _, used, logs, _ = workers.run()
    assert used == ["qwen3", "parakeet"] and len(spawned) == 2                      # 兩種模型各一個程序，第二個檔案沿用
    log = workers.model_log()
    assert log.count("load qwen") == 1 and log.count("load parakeet") == 1 and "load qwen Qwen/Qwen3-ASR-1.7B cuda bfloat16" in log
    assert log.count("qwen to cpu") == 2 and log.count("qwen to cuda") == 1        # 做完搬到 CPU，下一個檔案再搬回來
    assert any("沿用已載入的 Qwen3-ASR" in x for x in logs)


def test_a_missing_model_does_not_stop_the_others(workers, monkeypatch):
    monkeypatch.setenv("FAKE_ASR_MISSING", "parakeet")
    hyps, used, logs, _ = workers.run()
    assert used == ["qwen3"] and hyps[0] == {"qwen3": "今日は電気", "parakeet": None}
    assert any("Parakeet 失敗" in x and "nemo_toolkit[asr]" in x for x in logs)


def test_settings_reach_the_worker(workers, monkeypatch):
    backends = {k: dict(v) for k, v in config.CROSS_BACKENDS.items()}
    backends["parakeet"]["decoder"] = "ctc"
    backends["qwen3"]["use_hotwords"] = False
    backends["qwen3"]["python"] = sys.executable                                   # 指定別的 Python（虛擬環境）
    monkeypatch.setattr(config, "CROSS_BACKENDS", backends)
    monkeypatch.setattr(config, "CROSS_MODELS", ["qwen3", "parakeet", "whisperx"])
    spawned = []
    real_spawn = JOBS.spawn
    monkeypatch.setattr(JOBS, "spawn", lambda cmd, **kw: (spawned.append(cmd), real_spawn(cmd, **kw))[1])
    _, used, logs, _ = workers.run()
    assert used == ["qwen3", "parakeet"] and spawned[0][0] == sys.executable
    assert any("不認得的模型「whisperx」" in x for x in logs)
    log = workers.model_log()
    assert "context=''" in log and "parakeet decoder ctc" in log


def test_missing_worker_file_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ASR_WORKER", tmp_path / "nope.py")
    with pytest.raises(FileNotFoundError, match="nope.py"):
        cross_asr.run_cross_asr("/x/a.wav", SEGMENTS, [], lambda m: None, lambda p: None)


# ---------------- process_file 裡的交叉比對 ----------------
class Seg:
    def __init__(self, start, end, text, words=None):
        self.start, self.end, self.text, self.words = start, end, text, words
        self.no_speech_prob, self.avg_logprob = 0.0, -0.2


class W:
    def __init__(self, start, end, word):
        self.start, self.end, self.word = start, end, word


@pytest.fixture
def cross_stub(monkeypatch):
    calls = types.SimpleNamespace(aligned=[], cross=[])
    segments = [Seg(0, 2, "今日は天気", [W(0, 1, "今日は"), W(1, 2, "天気")]), Seg(3, 5, "こんにちは", [W(3, 5, "こんにちは")])]

    class Model:
        def transcribe(self, path, **kw):
            return iter(segments), types.SimpleNamespace(duration=10.0, duration_after_vad=9.0)

    def fake_cross(audio, segs, hotwords, log, progress):
        calls.cross.append((audio, [s["text"] for s in segs], hotwords))
        progress(0.5)
        progress(1.0)
        return calls.answer(segs)

    def fake_align(audio, segs, log, progress):
        calls.aligned.append([(s["text"], [w.word for w in s["words"]] if s.get("words") else None) for s in segs])
        progress(1.0)
        n = len(segs)
        return {"duration": 10.0, "spans": [[s["start"], s["end"]] for s in segs], "confs": [0.9] * n,
                "wide": [False] * n, "word_spans": [None] * n}

    calls.answer = lambda segs: ([{"qwen3": "今日は電気", "parakeet": "今日は電気"}, {"qwen3": "こんにちは", "parakeet": "こんにちは"}],
                                 ["qwen3", "parakeet"])
    monkeypatch.setattr(pipeline, "get_model", lambda log: Model())
    monkeypatch.setattr(pipeline, "run_cross_asr", fake_cross)
    monkeypatch.setattr(pipeline, "run_alignment", fake_align)

    def run(use_align=False, use_split=True, use_cross=True, hotwords=()):
        logs, prog = [], []
        out = pipeline.process_file("/x/影片.mp4", False, use_align, use_split, list(hotwords), logs.append, prog.append,
                                    add_blank=False, use_cross=use_cross)
        return Path(out), logs, [round(p, 4) for p in prog]
    calls.run = run
    return calls


def test_cross_check_fixes_the_text_before_alignment(cross_stub):
    out, logs, prog = cross_stub.run(use_align=True, hotwords=["クイズ"])
    assert out.name == "影片_large-v3_nosep_cross_align.srt"
    assert "今日は電気" in out.read_text(encoding="utf-8") and "天気" not in out.read_text(encoding="utf-8")
    assert cross_stub.cross[0] == ("/x/影片.mp4", ["今日は天気", "こんにちは"], ["クイズ"])
    assert cross_stub.aligned[0][0] == ("今日は電気", ["今日は", "電気"])            # 對齊用的是修正後的文字與詞
    assert prog == sorted(prog) and prog[-1] == 1.0
    assert prog[:4] == [0.13, 0.325, 0.725, 0.8]                                 # 轉錄 65% → 交叉比對 15% → 對齊 20%
    assert any("交叉比對 = Qwen3-ASR＋Parakeet" in x for x in logs)


def test_without_the_option_nothing_changes(cross_stub):
    out, _, prog = cross_stub.run(use_cross=False)
    assert out.name == "影片_large-v3_nosep.srt" and "天気" in out.read_text(encoding="utf-8") and not cross_stub.cross


def test_failed_cross_check_still_writes_whisper_text(cross_stub):
    def boom(segs):
        raise RuntimeError("顯存不足")
    cross_stub.answer = boom
    out, logs, _ = cross_stub.run()
    assert out.name == "影片_large-v3_nosep.srt" and "天気" in out.read_text(encoding="utf-8")
    assert any("[交叉比對] 失敗" in x and "顯存不足" in x for x in logs)


def test_with_only_one_model_differences_are_listed_but_not_fixed(cross_stub):
    cross_stub.answer = lambda segs: ([{"qwen3": "今日は電気", "parakeet": None}, {"qwen3": "こんにちは", "parakeet": None}], ["qwen3"])
    out, logs, _ = cross_stub.run()
    assert out.name == "影片_large-v3_nosep.srt" and "天気" in out.read_text(encoding="utf-8")
    assert any("只列出差異" in x for x in logs)
    assert Path(out).parent == Path(runtime.output_dir)
