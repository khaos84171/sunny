"""多模型交叉比對：投票規則（crosscheck.py）、套用到字幕與詞、日誌報告、每條字幕送去辨識的聲音範圍。"""
import pytest

from whisper_app import config, cross_asr, crosscheck as cc
from whisper_app.models import Word


def fixed(primary, q, p, protect=()):
    edits, held = cc.vote(primary, {"qwen3": q, "parakeet": p}, protect)
    return cc.apply_edits(primary, edits), [h.reason for h in held]


# ---------------- 比較用的字串 ----------------
def test_comparison_ignores_punctuation_width_kana_and_kanji_digits():
    assert cc.normalize_key("「ＱｕｉｚＫｎｏｃｋ」、３人。") == cc.normalize_key("quizknock 三人")
    assert cc.normalize_key("クイズ") == cc.normalize_key("くいず")
    keys, pos = cc.key_chars("あ、い")
    assert keys == ["あ", "い"] and pos == [0, 2]                      # 位置指回原文，標點不算


def test_match_positions_marks_only_identical_characters():
    m, dist = cc.match_positions(list("今日は天気"), list("今日は電気"))
    assert m == [0, 1, 2, None, 4] and dist == 1


# ---------------- 投票 ----------------
def test_two_models_agreeing_outvote_whisper_and_keep_whisper_punctuation():
    assert fixed("今日は天気がいいですね。", "今日は電気がいいですね", "今日は電気がいいですね。") == ("今日は電気がいいですね。", [])
    assert fixed("早押しクイズで、勝った！", "早押しクイズで言った", "早押しクイズで言った") == ("早押しクイズで、言った！", [])
    assert fixed("早押しクイズで勝った", "早押しクイズで買った", "早押しクイズで買った")[0] == "早押しクイズで勝った" or cc._JanomeTokenizer is None  # 同音：聽不出差別


def test_no_majority_keeps_whisper():
    assert fixed("今日は天気がいい", "今日は電気がいい", "今日は元気がいい") == ("今日は天気がいい", [])
    assert fixed("今日は天気がいい", "今日は電気がいい", "今日は天気がいい") == ("今日は天気がいい", [])


def test_missing_words_in_the_middle_are_inserted_and_extra_words_removed():
    assert fixed("これは大丈夫", "これは本当に大丈夫", "これは本当に大丈夫")[0] == "これは本当に大丈夫"
    assert fixed("これは本当に大丈夫、はい。", "これは大丈夫はい", "これは大丈夫、はい")[0] == "これは大丈夫、はい。"


@pytest.mark.parametrize("primary, other", [
    ("行きます", "行きます次の"),            # 結尾多出來的字：可能是聲音多包到下一句
    ("行きます", "えっと行きます"),          # 開頭多出來的字
    ("ありがとう。", "がとう"),              # 開頭少了：可能是聲音切得太緊
])
def test_edges_are_never_extended_or_trimmed(primary, other):
    assert fixed(primary, other, other) == (primary, ["edge"])


def test_fillers_are_not_added_or_removed():
    assert fixed("そう行くよ", "そうえー行くよ", "そうえー行くよ") == ("そう行くよ", ["filler"])
    assert fixed("そうあのー行くよ", "そう行くよ", "そう行くよ") == ("そうあのー行くよ", ["filler"])


def test_kanji_vs_kana_is_only_a_spelling_difference():
    assert fixed("一人で行きます", "ひとりで行きます", "ひとりで行きます") == ("一人で行きます", ["spelling"])


def test_same_reading_is_not_a_mishearing():
    pytest.importorskip("janome")
    assert fixed("クイズ王の伊沢です", "クイズ王の井沢です", "クイズ王の井沢です") == ("クイズ王の伊沢です", ["same_reading"])


def test_selected_hotwords_are_never_changed():
    assert fixed("伊沢拓司です", "石沢拓司です", "石沢拓司です", protect=["伊沢拓司"]) == ("伊沢拓司です", ["hotword"])
    assert fixed("伊沢拓司です", "石沢拓司です", "石沢拓司です")[0] == "石沢拓司です"


def test_large_disagreements_are_left_for_a_human(monkeypatch):
    monkeypatch.setattr(config, "CROSS_MAX_CHANGE_CHARS", 5)
    assert fixed("全然違う文章がここにあります", "まったく別のことを話しています", "まったく別のことを話しています") == \
        ("全然違う文章がここにあります", ["too_long"])


def test_one_model_alone_cannot_outvote_whisper():
    edits, held = cc.vote("今日は天気", {"qwen3": "今日は電気", "parakeet": None})
    assert edits == [] and held == []


def test_digits_and_kanji_numerals_count_as_the_same():
    assert fixed("3人で行く", "三人で行く", "三人で行く") == ("3人で行く", [])


# ---------------- 套用到詞（拆分與對齊要用）----------------
def test_edits_keep_word_boundaries_and_times():
    words = [Word(0, 1, " クイズ"), Word(1, 2, "王の"), Word(2, 3, "伊沢"), Word(3, 4, "です。")]
    surface = "".join(w.word for w in words)
    edits, _ = cc.vote(surface, {"a": "クイズ王の吉沢です", "b": "クイズ王の吉沢です"})
    assert cc.apply_edits_to_words(words, edits) == [
        Word(0, 1, " クイズ"), Word(1, 2, "王の"), Word(2, 3, "吉沢"), Word(3, 4, "です。")]


def test_edits_that_delete_a_whole_word_drop_it():
    words = [Word(0, 1, "これは"), Word(1, 2, "本当に"), Word(2, 3, "大丈夫")]
    edits, _ = cc.vote("これは本当に大丈夫", {"a": "これは大丈夫", "b": "これは大丈夫"})
    assert cc.apply_edits_to_words(words, edits) == [Word(0, 1, "これは"), Word(2, 3, "大丈夫")]


def test_inserted_text_joins_the_previous_word():
    words = [Word(0, 1, "これは"), Word(1, 2, "大丈夫")]
    edits, _ = cc.vote("これは大丈夫", {"a": "これは本当に大丈夫", "b": "これは本当に大丈夫"})
    assert cc.apply_edits_to_words(words, edits) == [Word(0, 1, "これは本当に"), Word(1, 2, "大丈夫")]


# ---------------- 套用到整份字幕 + 日誌 ----------------
LABELS = {"qwen3": "Qwen3-ASR", "parakeet": "Parakeet"}


def seg(start, text, words=None):
    return {"start": start, "end": start + 2, "text": text, "words": words}


def test_apply_crosscheck_fixes_text_and_words_and_logs_each_change():
    segs = [seg(0, "今日は天気がいい", [Word(0, 1, "今日は"), Word(1, 2, "天気が"), Word(2, 3, "いい")]),
            seg(5, "こんにちは")]
    hyps = [{"qwen3": "今日は電気がいい", "parakeet": "今日は電気がいい"},
            {"qwen3": "こんにちは", "parakeet": "こんにちは。"}]
    logs = []
    stats = cc.apply_crosscheck(segs, hyps, LABELS, logs.append)
    assert segs[0]["text"] == "今日は電気がいい" and segs[0]["words"][1] == Word(1, 2, "電気が") and segs[0]["cross_fixed"]
    assert segs[1]["text"] == "こんにちは" and not segs[1].get("cross_fixed")
    assert stats["fixed"] == 1 and stats["same"] == 1 and stats["edits"] == 1
    assert any("「天」→「電」" in x and "Qwen3-ASR、Parakeet 一致" in x for x in logs)
    assert any("2 條中 1 條各模型完全一致" in x for x in logs)


def test_subtitles_nobody_else_heard_are_reported_or_dropped(monkeypatch):
    hyps = [{"qwen3": "", "parakeet": ""}]
    logs = []
    segs = [seg(10, "ご視聴ありがとうございました")]
    assert cc.apply_crosscheck(segs, hyps, LABELS, logs.append)["unheard"] == 1
    assert len(segs) == 1 and any("聽不到任何字" in x and "00:00:10,000" in x for x in logs)
    monkeypatch.setattr(config, "CROSS_DROP_UNHEARD", True)
    logs = []
    stats = cc.apply_crosscheck(segs, hyps, LABELS, logs.append)
    assert segs == [] and stats["dropped"] == 1 and any("刪除" in x for x in logs)


def test_big_disagreement_without_majority_is_listed_for_review():
    logs = []
    cc.apply_crosscheck([seg(3, "全然違う内容")], [{"qwen3": "まったく別の話", "parakeet": "ほかのことです"}], LABELS, logs.append)
    assert any("建議檢查" in x for x in logs) and any("跟其他模型差很多" in x and "Qwen3-ASR：まったく別の話" in x for x in logs)


def test_failed_model_is_ignored_and_whisper_kept():
    segs = [seg(0, "今日は天気")]
    stats = cc.apply_crosscheck(segs, [{"qwen3": "今日は電気", "parakeet": None}], LABELS, lambda m: None)
    assert segs[0]["text"] == "今日は天気" and stats.get("fixed", 0) == 0 and segs[0]["cross"] == {"qwen3": "今日は電気"}


# ---------------- 送去辨識的聲音範圍 ----------------
def test_clip_spans_are_padded_but_stay_out_of_neighbours(monkeypatch):
    monkeypatch.setattr(config, "CROSS_PAD_SEC", 0.25)
    segs = [{"start": 0.1, "end": 2.0}, {"start": 2.1, "end": 4.0}, {"start": 6.0, "end": 9.9}]
    assert cross_asr.clip_spans(segs, 10.0) == [[0.0, 2.1], [2.0, 4.25], [5.75, 10.0]]
    overlapping = [{"start": 1.0, "end": 3.0}, {"start": 2.5, "end": 4.0}]
    assert cross_asr.clip_spans(overlapping, 10.0) == [[0.75, 3.0], [2.5, 4.25]]
