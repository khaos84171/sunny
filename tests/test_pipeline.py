"""完整處理流程：轉錄 → 幻覺過濾 → 對齊 → 拆分 → 輸出 SRT；進度條分配；輸出檔名；取消。"""
import threading
import time
import types
from pathlib import Path

import pytest

from whisper_app import config, pipeline, runtime
from whisper_app.jobs import JOBS, JobCancelled


class Seg:
    def __init__(self, start, end, text, no_speech=0.0, logprob=-0.3):
        self.start, self.end, self.text, self.words = start, end, text, None
        self.no_speech_prob, self.avg_logprob = no_speech, logprob


class FakeModel:
    def __init__(self, segments=None):
        self.kwargs = []
        self.segments = segments if segments is not None else [
            Seg(0, 10, "こんにちは"), Seg(10, 20, PHRASE := "ご視聴ありがとうございました", no_speech=0.7), Seg(20, 30, PHRASE + "。", no_speech=0.01, logprob=-0.2)]

    def transcribe(self, path, **kw):
        self.kwargs.append(kw)
        return iter(self.segments), types.SimpleNamespace(duration=100.0, duration_after_vad=90.0)


def fake_align(audio, segments, log, progress, detect_offset=False):
    progress(0.5)
    progress(1.0)
    n = len(segments)
    return {"duration": 100.0, "offset": 0.0, "spans": [[s["start"], s["end"]] for s in segments], "confs": [0.9] * n,
            "wide": [False] * n, "word_spans": [None] * n}


@pytest.fixture
def stub(monkeypatch):
    model = FakeModel()
    calls = types.SimpleNamespace(separation=[])

    def fake_separate(path, work, log, **kw):
        calls.separation.append((path, work, kw))
        for f in (0.0, 0.5, 1.0):
            kw["progress_func"](f)
        return "/x/vocals.wav"
    monkeypatch.setattr(pipeline, "get_model", lambda log: model)
    monkeypatch.setattr(pipeline, "run_alignment", fake_align)
    monkeypatch.setattr(pipeline, "separate_vocals", fake_separate)
    stub = types.SimpleNamespace(model=model, calls=calls)

    def run(use_sep=False, use_align=False, use_split=True, hotwords=(), add_blank=False):
        logs, prog = [], []
        out = pipeline.process_file("/x/影片.mp4", use_sep, use_align, use_split, list(hotwords), logs.append, prog.append, add_blank=add_blank)
        return out, logs, [round(p, 4) for p in prog]
    stub.run = run
    return stub


# ---------------- 拆分與跨片段合併 ----------------
def seg_with_words(start, end, text, tokens):
    """tokens = [(詞, 起, 迄), ...]"""
    seg = Seg(start, end, text)
    seg.words = [types.SimpleNamespace(start=a, end=b, word=w) for w, a, b in tokens]
    return seg


def test_sentence_split_across_two_whisper_segments_is_merged(stub, monkeypatch):
    monkeypatch.setattr(config, "EXTEND_END_TO_NEXT_SEC", 0)
    stub.model.segments = [
        seg_with_words(0.0, 1.0, "でもペアの強みって", [("でも", 0.0, 0.3), ("ペア", 0.3, 0.6), ("の", 0.6, 0.7), ("強み", 0.7, 0.9), ("って", 0.9, 1.0)]),
        seg_with_words(1.2, 2.5, "押した後だと思うんですよ。", [("押した", 1.2, 1.6), ("後", 1.6, 1.8), ("だ", 1.8, 1.9), ("と", 1.9, 2.0), ("思う", 2.0, 2.2), ("んです", 2.2, 2.4), ("よ", 2.4, 2.45), ("。", 2.45, 2.5)]),
        seg_with_words(3.0, 4.0, "次の問題です。", [("次の", 3.0, 3.4), ("問題", 3.4, 3.8), ("です", 3.8, 3.95), ("。", 3.95, 4.0)]),
    ]
    out, logs, _ = stub.run()
    blocks = Path(out).read_text(encoding="utf-8").strip().split("\n\n")
    assert [b.split("\n", 2)[2] for b in blocks] == ["でもペアの強みって押した後だと思うんですよ。", "次の問題です。"]
    assert blocks[0].split("\n")[1] == "00:00:00,000 --> 00:00:02,500"
    assert any(x.startswith("[合併]") and "でもペアの強みって" in x for x in logs)
    assert any("Whisper 原本 3 段 → 拆成 2 條字幕" in x for x in logs)


def test_subtitle_end_is_extended_to_the_next_start_when_the_gap_is_small(stub):
    stub.model.segments = [
        seg_with_words(0.0, 1.0, "一つ目です。", [("一つ目", 0.0, 0.8), ("です", 0.8, 0.95), ("。", 0.95, 1.0)]),
        seg_with_words(2.5, 3.5, "二つ目です。", [("二つ目", 2.5, 3.3), ("です", 3.3, 3.45), ("。", 3.45, 3.5)]),     # 空隙 1.5 秒 → 延長
        seg_with_words(6.0, 7.0, "三つ目です。", [("三つ目", 6.0, 6.8), ("です", 6.8, 6.95), ("。", 6.95, 7.0)]),     # 空隙 2.5 秒 → 不動
    ]
    out, _, _ = stub.run()
    times = [b.split("\n")[1] for b in Path(out).read_text(encoding="utf-8").strip().split("\n\n")]
    assert times == ["00:00:00,000 --> 00:00:02,500", "00:00:02,500 --> 00:00:03,500", "00:00:06,000 --> 00:00:07,000"]


def test_end_extension_can_be_turned_off(stub, monkeypatch):
    monkeypatch.setattr(config, "EXTEND_END_TO_NEXT_SEC", None)
    stub.model.segments = [
        seg_with_words(0.0, 1.0, "一つ目です。", [("一つ目", 0.0, 0.8), ("です", 0.8, 0.95), ("。", 0.95, 1.0)]),
        seg_with_words(2.5, 3.5, "二つ目です。", [("二つ目", 2.5, 3.3), ("です", 3.3, 3.45), ("。", 3.45, 3.5)]),
    ]
    out, _, _ = stub.run()
    assert Path(out).read_text(encoding="utf-8").count("00:00:01,000") == 1


def test_merge_can_be_turned_off(stub, monkeypatch):
    monkeypatch.setattr(config, "MERGE_FRAGMENTS", False)
    stub.model.segments = [
        seg_with_words(0.0, 0.5, "はい", [("はい", 0.0, 0.5)]),
        seg_with_words(0.6, 1.0, "ー。", [("ー", 0.6, 0.9), ("。", 0.9, 1.0)]),
    ]
    out, logs, _ = stub.run()
    assert Path(out).read_text(encoding="utf-8").count("-->") == 2 and not any(x.startswith("[合併]") for x in logs)


def test_alignment_report_numbers_follow_merged_subtitles(stub, monkeypatch):
    """合併過的字幕涵蓋好幾個 Whisper 片段：對齊報告的編號要對得上輸出的 SRT。"""
    monkeypatch.setattr(pipeline, "run_alignment", lambda *a, **k: {
        **fake_align(None, [{"start": 0, "end": 0}] * 3, None, lambda f: None),
        "confs": [0.9, 0.05, 0.9], "spans": [[0.0, 0.5], [0.6, 1.0], [3.0, 4.0]], "word_spans": [None] * 3})
    stub.model.segments = [
        seg_with_words(0.0, 0.5, "はい", [("はい", 0.0, 0.5)]),
        seg_with_words(0.6, 1.0, "ー。", [("ー", 0.6, 0.9), ("。", 0.9, 1.0)]),
        seg_with_words(3.0, 4.0, "次です。", [("次", 3.0, 3.4), ("です", 3.4, 3.9), ("。", 3.9, 4.0)]),
    ]
    out, logs, _ = stub.run(use_align=True)
    assert Path(out).read_text(encoding="utf-8").strip().count("-->") == 2
    assert any("#1 [" in x and "信心 0.05" in x for x in logs)      # 第二個片段（信心低）併進了 SRT 第 1 條


# ---------------- 幻覺過濾與轉錄選項 ----------------
def test_hallucinated_subtitle_is_dropped_and_reported(stub):
    out, logs, prog = stub.run()
    text = Path(out).read_text(encoding="utf-8")
    assert text.count("-->") == 2 and "こんにちは" in text and text.count("ご視聴ありがとうございました") == 1 and "00:00:20,000" in text
    hits = [x for x in logs if "幻覺過濾" in x]
    assert len(hits) == 1 and "0.70" in hits[0] and "00:00:10,000" in hits[0]
    assert prog[-1] == 1.0


def test_transcribe_options_reach_whisper(stub):
    stub.run(hotwords=["クイズ", "東大"])
    kw = stub.model.kwargs[-1]
    assert kw["language"] == "ja" and kw["hallucination_silence_threshold"] == 2.0
    assert kw["word_timestamps"] is True and kw["hotwords"] == "クイズ、東大。"
    stub.run(use_split=False)
    assert stub.model.kwargs[-1]["word_timestamps"] is False and stub.model.kwargs[-1]["hotwords"] is None


# ---------------- 輸出檔名 ----------------
@pytest.mark.parametrize("sep, align, split, name", [
    (False, False, True, "影片_large-v3_nosep.srt"),                 # 預設（有拆分）：檔名跟以前一樣
    (True, True, True, "影片_large-v3_sep_align.srt"),
    (False, False, False, "影片_large-v3_nosep_nosplit.srt"),         # 沒拆分：多一個標記，不蓋掉有拆分的版本
    (True, True, False, "影片_large-v3_sep_align_nosplit.srt"),
    (False, True, False, "影片_large-v3_nosep_align_nosplit.srt"),
])
def test_output_file_names(stub, sep, align, split, name):
    out, _, _ = stub.run(sep, align, split)
    assert Path(out).name == name and Path(out).parent == Path(runtime.output_dir)


def test_output_name_says_when_the_sound_refinement_is_off(stub, monkeypatch):
    """關掉聲音微調（ALIGN_REFINE = False）的輸出加上 _ctc，不會蓋掉微調版，方便拿去跟標準答案比較。"""
    monkeypatch.setattr(config, "ALIGN_REFINE", False)
    assert Path(stub.run(True, True, True)[0]).name == "影片_large-v3_sep_align_ctc.srt"
    assert Path(stub.run(False, True, False)[0]).name == "影片_large-v3_nosep_align_ctc_nosplit.srt"
    assert Path(stub.run(False, False, True)[0]).name == "影片_large-v3_nosep.srt"       # 沒做對齊就沒有這個標記


def test_failed_alignment_still_writes_an_unaligned_file(stub, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("對齊壞了")
    monkeypatch.setattr(pipeline, "run_alignment", boom)
    out, logs, _ = stub.run(use_align=True)
    assert Path(out).name == "影片_large-v3_nosep.srt" and any("失敗" in x and "對齊壞了" in x for x in logs)


def test_leading_blank_subtitle_is_added_on_request(stub):
    stub.model.segments[0].start = 2.0                      # 第一句不是從影片開頭就開始
    out, _, _ = stub.run(add_blank=True)
    first = Path(out).read_text(encoding="utf-8").split("\n\n")[0]
    assert first == "1\n00:00:00,000 --> 00:00:02,000\n\u200b"
    out, _, _ = stub.run(add_blank=False)
    assert "\u200b" not in Path(out).read_text(encoding="utf-8")

# ---------------- 進度條 ----------------
def test_progress_without_separation_is_unscaled(stub):
    _, _, prog = stub.run(use_align=True)
    assert prog[0] == pytest.approx(0.1 * 0.8) and prog[-1] == 1.0 and prog == sorted(prog)   # 轉錄佔前 80%，對齊佔後 20%


def test_separation_takes_the_first_quarter_of_the_progress_bar(stub):
    _, _, prog = stub.run(use_sep=True, use_align=True)
    share = config.SEPARATION_PROGRESS_SHARE
    assert prog[:3] == [0.0, round(share / 2, 4), share] and prog[3] > share      # 分離：0 → 25%，之後才是轉錄
    assert prog == sorted(prog) and prog[-1] == 1.0


def test_separation_is_asked_with_configured_device_and_model(stub):
    stub.run(use_sep=True)
    path, work, kw = stub.calls.separation[0]
    assert kw["device"] == config.SEPARATION_DEVICE and kw["model_name"] == config.SEPARATION_MODEL
    assert work == runtime.separation_work_dir


# ---------------- 只對齊現有 SRT ----------------
def test_align_only_progress_and_output_name(monkeypatch, tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nこんにちは\n\n2\n00:00:04,000 --> 00:00:06,000\nさようなら\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "run_alignment", fake_align)
    prog = []
    out = pipeline.align_existing_srt(str(srt), "/x/影片.mp4", lambda m: None, prog.append)
    assert Path(out).name == "a_align.srt" and prog[0] == pytest.approx(0.5) and prog[-1] == 1.0
    assert prog == sorted(prog)
    monkeypatch.setattr(config, "ALIGN_REFINE", False)
    assert Path(pipeline.align_existing_srt(str(srt), "/x/影片.mp4", lambda m: None, lambda p: None)).name == "a_align_ctc.srt"


def test_align_only_never_separates_vocals(monkeypatch, tmp_path):
    """只對齊直接用原始影片音訊，不會呼叫 Demucs（不管視窗上有沒有勾人聲分離）。"""
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nこんにちは\n", encoding="utf-8")
    aligned_inputs = []

    def recording_align(audio, segments, log_func, progress_func, detect_offset=False):
        aligned_inputs.append(audio)
        return fake_align(audio, segments, log_func, progress_func, detect_offset)

    def no_separation(*a, **k):
        raise AssertionError("只對齊不該做人聲分離")
    monkeypatch.setattr(pipeline, "run_alignment", recording_align)
    monkeypatch.setattr(pipeline, "separate_vocals", no_separation)
    pipeline.align_existing_srt(str(srt), "/x/影片.mp4", lambda m: None, lambda p: None)
    assert aligned_inputs == ["/x/影片.mp4"]


def test_align_only_asks_for_offset_detection_and_keeps_the_srt_timeline(monkeypatch, tmp_path):
    """剪輯軟體匯出的字幕從 01:00:00 開始：對齊完仍然是那條時間軸，空白字幕也從 01:00:00 開始。"""
    srt = tmp_path / "a.srt"
    srt.write_text("1\n01:00:02,000 --> 01:00:04,000\nこんにちは\n\n2\n01:00:06,000 --> 01:00:08,000\nさようなら\n", encoding="utf-8")
    seen = {}

    def align_with_offset(audio, segments, log, progress, detect_offset=False):
        seen["detect_offset"] = detect_offset
        n = len(segments)
        return {"duration": 100.0 + 3600.0, "offset": 3600.0, "spans": [[s["start"] + 0.1, s["end"] - 0.1] for s in segments],
                "confs": [0.9] * n, "wide": [False] * n, "word_spans": [None] * n}
    monkeypatch.setattr(pipeline, "run_alignment", align_with_offset)
    out = pipeline.align_existing_srt(str(srt), "/x/影片.mp4", lambda m: None, lambda p: None, add_blank=True)
    text = Path(out).read_text(encoding="utf-8")
    assert seen["detect_offset"] is True
    assert text.startswith("1\n01:00:00,000 --> 01:00:02,100\n\u200b\n\n2\n01:00:02,100 --> ")
    assert "01:00:06,100" in text and text.count("-->") == 3


def test_align_only_rejects_srt_without_subtitles(tmp_path):
    srt = tmp_path / "empty.srt"
    srt.write_text("這不是字幕", encoding="utf-8")
    with pytest.raises(ValueError, match="沒有讀到任何字幕"):
        pipeline.align_existing_srt(str(srt), "/x/影片.mp4", lambda m: None, lambda f: None)


# ---------------- 取消 ----------------
def test_cancelled_job_writes_no_file(stub, monkeypatch):
    class CancellingModel(FakeModel):
        def transcribe(self, path, **kw):
            def gen():
                yield Seg(0, 10, "こんにちは")
                JOBS.cancel()                                    # 使用者在轉錄中途按了取消
                yield Seg(10, 20, "もう一句")
            return gen(), types.SimpleNamespace(duration=100.0, duration_after_vad=90.0)
    monkeypatch.setattr(pipeline, "get_model", lambda log: CancellingModel())
    JOBS.begin(JOBS.generation)
    with pytest.raises(JobCancelled):
        stub.run()
    assert not list(Path(runtime.output_dir).glob("*.srt"))


def test_cancellation_is_not_swallowed_as_an_alignment_failure(stub, monkeypatch):
    def cancelled(*a, **k):
        raise JobCancelled()
    monkeypatch.setattr(pipeline, "run_alignment", cancelled)
    with pytest.raises(JobCancelled):
        stub.run(use_align=True)


@pytest.mark.parametrize("stuck_at", ["載入模型", "解碼音訊與 VAD", "解碼一個視窗"])
def test_cancel_does_not_wait_for_whisper_wherever_it_is_stuck(stub, monkeypatch, stuck_at):
    """Whisper 在主程式裡跑、殺不掉：卡在哪一步，按取消都要馬上停下來，之後也不能再寫日誌或輸出檔案。"""
    release = threading.Event()
    info = types.SimpleNamespace(duration=100.0, duration_after_vad=90.0)

    class StuckModel(FakeModel):
        def transcribe(self, path, **kw):
            if stuck_at == "解碼音訊與 VAD":
                release.wait(10)

            def gen():
                yield Seg(0, 10, "こんにちは")
                if stuck_at == "解碼一個視窗":
                    release.wait(10)
                yield Seg(10, 20, "もう一句")
            return gen(), info

    def get_model(log):
        if stuck_at == "載入模型":
            release.wait(10)
        return StuckModel()
    monkeypatch.setattr(pipeline, "get_model", get_model)
    logs = []
    JOBS.begin(JOBS.generation)
    threading.Timer(0.3, JOBS.cancel).start()
    t0 = time.time()
    try:
        with pytest.raises(JobCancelled):
            pipeline.process_file("/x/影片.mp4", False, False, True, [], logs.append, lambda f: None, add_blank=False)
        assert time.time() - t0 < 2                        # 沒有等到 release：不必等 Whisper 做完
    finally:
        release.set()
    time.sleep(0.3)                                        # 讓背景那邊做完，確認它不會再影響到已取消的這個檔案
    assert not any("もう一句" in x for x in logs)
    assert not list(Path(runtime.output_dir).glob("*.srt"))
