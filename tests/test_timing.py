"""對齊後的時間處理：套用對齊結果、產生檢查報告、最後的時間微調。"""
import copy
import random

import pytest

from whisper_app import config, timing
from whisper_app.models import Word


def subs_from(*spans):
    return [{"start": s, "end": e, "text": "x", "aligned": True} for s, e in spans]


def as_pairs(subs):
    return [(round(s["start"], 3), round(s["end"], 3)) for s in subs]


# ---------------- finalize_aligned_timing ----------------
def test_short_subtitle_does_not_push_back_the_next_one():
    """「はい」1.00–1.10，下一句 1.20 開始：舊版會把下一句推到 1.30。"""
    subs = subs_from((1.00, 1.10), (1.20, 2.50))
    timing.finalize_aligned_timing(subs, 10.0)
    assert as_pairs(subs) == [(1.0, 1.2), (1.2, 2.75)]


def test_end_hold_and_min_duration_stack_when_there_is_room():
    subs = subs_from((1.0, 1.1), (5.0, 6.0))
    timing.finalize_aligned_timing(subs, 10.0)
    assert subs[0]["end"] == pytest.approx(max(1.1, 1.0 + config.ALIGN_MIN_DURATION) + config.ALIGN_END_HOLD_SEC)
    assert subs[1]["end"] == pytest.approx(6.0 + config.ALIGN_END_HOLD_SEC)


def test_last_subtitle_is_limited_by_the_media_duration():
    subs = subs_from((9.8, 9.9))
    timing.finalize_aligned_timing(subs, 10.0)
    assert subs[0]["end"] == pytest.approx(10.0)


def test_fully_covered_subtitle_still_has_positive_length_and_no_overlap():
    subs = subs_from((1.0, 3.0), (2.0, 2.2), (2.5, 4.0))
    timing.finalize_aligned_timing(subs, 10.0)
    assert all(s["end"] > s["start"] for s in subs)
    assert all(a["end"] <= b["start"] + 1e-9 for a, b in zip(subs, subs[1:]))


def test_start_lead_moves_aligned_subtitles_earlier(monkeypatch):
    monkeypatch.setattr(config, "ALIGN_START_LEAD_SEC", 0.1)
    subs = subs_from((2.0, 3.0))
    subs.append({"start": 5.0, "end": 6.0, "text": "unaligned", "aligned": False})
    timing.finalize_aligned_timing(subs, 10.0)
    assert subs[0]["start"] == pytest.approx(1.9) and subs[1]["start"] == 5.0


def test_properties_hold_for_random_inputs():
    rng = random.Random(1)
    for _ in range(500):
        t, subs = 0.0, []
        for _ in range(rng.randint(1, 8)):
            t += rng.choice([0.0, 0.05, 0.2, 0.7, 1.5])
            d = rng.choice([0.05, 0.15, 0.4, 1.0, 2.5])
            subs.append({"start": t, "end": t + d, "text": "x", "aligned": True})
            t += d
        original = copy.deepcopy(subs)
        timing.finalize_aligned_timing(subs, t + 5)
        for i, (new, old) in enumerate(zip(subs, original)):
            assert new["end"] >= new["start"]
            assert new["end"] >= old["end"] - 1e-9                       # 只會延長，不會縮短
            if i + 1 < len(subs):
                assert subs[i + 1]["start"] >= new["end"] - 1e-9         # 不重疊
                if old["end"] <= original[i + 1]["start"]:               # 本來就沒重疊 → 下一條的起點一點都不動
                    assert subs[i + 1]["start"] == pytest.approx(original[i + 1]["start"])


# ---------------- apply_alignment ----------------
def make_segment(**kw):
    seg = {"start": 1.0, "end": 3.0, "text": "今日は", "words": [Word(1.0, 2.0, "今日"), Word(2.0, 3.0, "は")]}
    seg.update(kw)
    return seg


def test_apply_alignment_updates_times_and_uses_worker_word_times():
    segs = [make_segment()]
    result = {"spans": [[1.2, 2.6]], "confs": [0.8], "wide": [True], "word_spans": [[[1.2, 1.9], [2.1, 2.6]]]}
    logs = []
    timing.apply_alignment(segs, result, logs.append)
    seg = segs[0]
    assert (seg["start"], seg["end"]) == (1.2, 2.6) and seg["orig_start"] == 1.0
    assert seg["conf"] == 0.8 and seg["wide"] is True and seg["aligned"] is True
    assert [(w.start, w.end, w.word) for w in seg["words"]] == [(1.2, 1.9, "今日"), (2.1, 2.6, "は")]
    assert "1/1" in logs[0]


def test_apply_alignment_unaligned_segment_keeps_original_times():
    segs = [make_segment()]
    timing.apply_alignment(segs, {"spans": [None], "confs": [None], "wide": [False], "word_spans": [None]}, lambda m: None)
    assert (segs[0]["start"], segs[0]["end"]) == (1.0, 3.0) and segs[0]["aligned"] is False


def test_apply_alignment_punctuation_word_sticks_to_previous_word():
    segs = [make_segment(words=[Word(1, 2, "今日"), Word(2, 2, "。")])]
    timing.apply_alignment(segs, {"spans": [[1.0, 2.0]], "confs": [0.9], "wide": [False], "word_spans": [[[1.0, 1.8], None]]}, lambda m: None)
    assert (segs[0]["words"][1].start, segs[0]["words"][1].end) == (1.8, 1.8)


def test_apply_alignment_without_word_times_rescales_whisper_words():
    """舊版對齊程序沒有傳回詞時間：把 Whisper 的詞時間等比例搬進對齊後的範圍。"""
    segs = [make_segment()]
    timing.apply_alignment(segs, {"spans": [[10.0, 12.0]], "confs": [0.9], "wide": [False], "word_spans": [None]}, lambda m: None)
    words = segs[0]["words"]
    assert (words[0].start, words[0].end, words[1].end) == (10.0, 11.0, 12.0)


def test_apply_alignment_keeps_the_sound_check_of_each_word():
    segs = [make_segment(), make_segment()]
    checks = [[["moved", -0.06, "ok", 0.01], ["ok", 0.0, "moved", 0.09]], None]
    result = {"spans": [[1.0, 2.0], [3.0, 4.0]], "confs": [0.8, 0.8], "wide": [False, False],
              "word_spans": [[[1.0, 1.5], [1.5, 2.0]], [[3.0, 3.5], [3.5, 4.0]]], "checks": checks}
    timing.apply_alignment(segs, result, lambda m: None)
    assert segs[0]["checks"] == checks[0] and segs[1]["checks"] is None


def test_apply_alignment_accepts_a_result_from_an_old_worker_without_checks():
    segs = [make_segment()]
    timing.apply_alignment(segs, {"spans": [[1.2, 2.6]], "confs": [0.8], "wide": [False], "word_spans": [None]}, lambda m: None)
    assert segs[0]["checks"] is None and segs[0]["aligned"] is True


# ---------------- detect_timeline_offset ----------------
def lines_at(*starts):
    return [{"start": s, "end": s + 1.5, "text": "x"} for s in starts]


def test_offset_is_zero_for_a_normal_subtitle_file():
    assert timing.detect_timeline_offset(lines_at(1, 30, 200, 1000), duration=1200) == 0.0


def test_editor_timeline_starting_at_one_hour_is_detected():
    """DaVinci Resolve／Final Cut 的時間軸預設從 01:00:00:00 開始，匯出的字幕整個晚了一小時。"""
    assert timing.detect_timeline_offset(lines_at(3600 + 3.9, 3600 + 63.9, 3600 + 1138), duration=1200) == 3600.0


def test_other_whole_hour_starts_are_detected_too():
    assert timing.detect_timeline_offset(lines_at(36000 + 5, 36000 + 300), duration=600) == 36000.0
    assert timing.detect_timeline_offset(lines_at(7200 + 5, 7200 + 300), duration=600) == 7200.0


def test_the_largest_fitting_whole_hour_is_used():
    """從 02:00:00 開始的字幕、影片有 66 分鐘：扣掉 1 小時（落在 60～66 分）和扣掉 2 小時（落在 0～5 分）都放得進去，
    字幕從影片最開頭附近開始才合理，所以取最大的。"""
    assert timing.detect_timeline_offset(lines_at(7200 + 10, 7200 + 100, 7200 + 300), duration=4000) == 7200.0
    assert timing.detect_timeline_offset(lines_at(3600 + 10, 3600 + 3000, 3600 + 4000), duration=4500) == 3600.0   # 影片較長：只有 1 小時放得下


def test_a_video_that_really_is_long_is_not_shifted():
    """影片真的有 2 小時，字幕從 1 小時之後才開始講話（前面沒人說話）：不是偏移，不能亂扣。"""
    assert timing.detect_timeline_offset(lines_at(3700, 4000, 6000), duration=7200) == 0.0


def test_a_few_lines_past_the_end_do_not_trigger_a_shift():
    starts = list(range(10, 1000, 10)) + [5000, 5010]                # 影片被截短了：少數字幕超過影片長度
    assert timing.detect_timeline_offset(lines_at(*starts), duration=1000) == 0.0


def test_mostly_inside_but_trimmed_video_is_processed_normally():
    """一半以上還落在影片裡（影片被截短）：照常處理，不丟錯。"""
    starts = list(range(10, 1000, 10))[:60] + [5000 + i for i in range(30)]
    assert timing.detect_timeline_offset(lines_at(*starts), duration=1000) == 0.0


def test_subtitles_that_cannot_be_matched_to_the_video_raise_a_helpful_error():
    with pytest.raises(ValueError) as e:
        timing.detect_timeline_offset(lines_at(5000, 5100, 5200, 5300), duration=600)      # 差 1 小時多，不是整數小時
    message = str(e.value)
    assert "01:23:20,000" in message and "00:10:00,000" in message and "對不起來" in message and "00:59:50" in message


def test_subtitles_longer_than_the_video_even_after_shifting_are_rejected():
    with pytest.raises(ValueError):
        timing.detect_timeline_offset(lines_at(3600, 3700, 3800, 3900, 4000), duration=100)


def test_offset_detection_with_nothing_to_align():
    assert timing.detect_timeline_offset([], duration=100) == 0.0


# ---------------- report_alignment ----------------
def test_report_lists_unaligned_low_confidence_and_big_shifts(monkeypatch):
    segs = [
        {"start": 0, "end": 1, "text": "ok", "conf": 0.9, "aligned": True, "orig_start": 0.0},
        {"start": 2, "end": 3, "text": "沒對上", "conf": None, "aligned": False, "orig_start": 2.0},
        {"start": 4, "end": 5, "text": "信心低", "conf": 0.1, "aligned": True, "orig_start": 4.0},
        {"start": 9, "end": 10, "text": "移很多", "conf": 0.9, "aligned": True, "orig_start": 6.0, "wide": True},
        {"start": 11, "end": 12, "text": "  ", "conf": None, "aligned": False, "orig_start": 11.0},      # 空白字幕不列
    ]
    subs = [{"src": i, "start": s["start"], "end": s["end"], "text": s["text"]} for i, s in enumerate(segs)]
    subs.insert(0, {"start": 0, "end": 0, "text": "​"})                                            # 開頭空白字幕：沒有 src，編號往後推
    logs = []
    timing.report_alignment(segs, subs, logs.append)
    text = "\n".join(logs)
    assert "3 處" in text
    assert "#3" in text and "對不上" in text and "#4" in text and "信心 0.10" in text and "#5" in text and "大範圍重對" in text
    assert "ok" not in text.replace("okay", "")


def test_report_lists_words_placed_where_there_is_no_sound(monkeypatch):
    """聲音交叉檢查：CTC 說有字、該處卻聽不到聲音（Whisper 幻覺或對到錯的地方）→ 列出是哪個詞。"""
    fine = ["ok", 0.0, "ok", 0.0]
    segs = [
        {"start": 0, "end": 1, "text": "正常", "conf": 0.9, "aligned": True, "orig_start": 0.0, "checks": [fine],
         "words": [Word(0, 1, "正常")]},
        {"start": 2, "end": 4, "text": "今日は誰もいない", "conf": 0.9, "aligned": True, "orig_start": 2.0,
         "checks": [fine, ["silent", None, "silent", None]], "words": [Word(2, 3, "今日は"), Word(3, 4, "誰もいない")]},
        {"start": 5, "end": 6, "text": "分不出來", "conf": 0.9, "aligned": True, "orig_start": 5.0,
         "checks": [["noisy", None, "noisy", None]], "words": [Word(5, 6, "分不出來")]},      # 無法判定不算有問題
        {"start": 7, "end": 8, "text": "沒有詞時間", "conf": 0.9, "aligned": True, "orig_start": 7.0,
         "checks": [["ok", 0.0, "silent", None]]},                                            # 沒有 words 時用整句
    ]
    subs = [{"src": i, "start": s["start"], "end": s["end"], "text": s["text"]} for i, s in enumerate(segs)]
    logs = []
    timing.report_alignment(segs, subs, logs.append)
    text = "\n".join(logs)
    assert "2 處" in text
    assert "#2" in text and "「誰もいない」的位置聽不到聲音" in text
    assert "#4" in text and "「沒有詞時間」的位置聽不到聲音" in text
    assert "#1" not in text and "#3" not in text


def test_report_prefers_the_more_serious_reason_when_several_apply():
    seg = {"start": 2, "end": 3, "text": "信心低又聽不到", "conf": 0.1, "aligned": True, "orig_start": 2.0,
           "checks": [["silent", None, "silent", None]], "words": [Word(2, 3, "信心低又聽不到")]}
    logs = []
    timing.report_alignment([seg], [{"src": 0, "start": 2, "end": 3, "text": "x"}], logs.append)
    assert len(logs) == 2 and "信心 0.10" in logs[1] and "聽不到聲音" not in logs[1]


def test_report_is_silent_when_everything_is_fine():
    segs = [{"start": 0, "end": 1, "text": "ok", "conf": 0.9, "aligned": True, "orig_start": 0.0}]
    logs = []
    timing.report_alignment(segs, [{"src": 0, "start": 0, "end": 1, "text": "ok"}], logs.append)
    assert logs == []
