"""字幕拆分：句尾標點、停頓、長度上限、右括號、不切在詞中間。有 janome 與沒有（簡易規則）兩種模式都測。"""
import pytest

from whisper_app import config, splitter
from whisper_app.models import Word


def words(*texts, gaps=None):
    """依序排好的詞，每個 0.4 秒；gaps = {第幾個詞後面: 停頓秒數}，沒指定的停 0.05 秒。"""
    out, t = [], 0.0
    for i, text in enumerate(texts):
        out.append(Word(t, t + 0.4, text))
        t += 0.4 + (gaps or {}).get(i, 0.05)
    return out


def pieces(ws):
    return [p["text"] for p in splitter.split_segment({"start": 0, "end": 99, "text": "", "words": ws})]


@pytest.fixture(params=["janome", "簡易規則"])
def mode(request, monkeypatch):
    if request.param == "janome":
        pytest.importorskip("janome")
    else:
        monkeypatch.setattr(splitter, "_JanomeTokenizer", None)
    return request.param


def test_no_words_returns_the_segment_unchanged():
    seg = {"start": 1.0, "end": 2.0, "text": "abc", "words": None}
    assert splitter.split_segment(seg) == [{"start": 1.0, "end": 2.0, "text": "abc"}]


def test_splits_after_sentence_end_punctuation(mode):
    assert pieces(words("ありがとう", "。", "次の", "文です")) == ["ありがとう。", "次の文です"]


def test_splits_on_a_clear_pause_at_a_phrase_boundary(mode):
    assert pieces(words("こんにちは", "みなさん", gaps={0: 1.0})) == ["こんにちは", "みなさん"]


def test_never_cuts_inside_a_word_even_with_a_pause(mode):
    """「熊本地震」は熊｜本 の間に長い停頓があっても切らない（漢字どうしは同じ語）。"""
    assert pieces(words("熊", "本", "地震", gaps={0: 0.6, 1: 0.6})) == ["熊本地震"]


def test_too_long_segment_is_cut_at_commas(mode):
    ws = words(*(["あいうえお、"] * 8))                       # 48 字，超過 SPLIT_MAX_CHARS
    result = pieces(ws)
    assert len(result) >= 2 and "".join(result) == "".join(w.word for w in ws)
    assert all(len(p) <= config.SPLIT_MAX_CHARS for p in result)


def test_pieces_keep_order_and_timing(mode):
    ws = words("ありがとう", "。", "次の", "文です")
    result = splitter.split_segment({"start": 0, "end": 99, "text": "", "words": ws})
    assert result[0]["start"] == ws[0].start and result[0]["end"] == ws[1].end
    assert result[1]["start"] == ws[2].start and result[1]["end"] == ws[3].end


# ---- 句尾標點後面接右括號／引號 ----
@pytest.mark.parametrize("tokens, expected", [
    (("「", "ありがとう", "。", "」", "次の", "文です"), ["「ありがとう。」", "次の文です"]),      # 。 和 」 是兩個詞
    (("「", "ありがとう", "。」", "次の", "文です"), ["「ありがとう。」", "次の文です"]),         # 。」 黏成一個詞
    (("え", "っ", "？", "』", "）", "本当", "です", "か"), ["えっ？』）", "本当ですか"]),            # 連續多個括號
    (("ありがとう", "。", "「", "次の", "文です"), ["ありがとう。", "「次の文です"]),               # 開頭引號跟著下一條
    (("ありがとう", "。", "次の", "文です"), ["ありがとう。", "次の文です"]),                       # 沒有括號：行為不變
])
def test_closing_brackets_stay_with_their_sentence(mode, tokens, expected):
    assert pieces(words(*tokens)) == expected


def test_sentence_end_flags_move_to_the_last_closing_bracket():
    flags = splitter._sentence_end_flags(words("ありがとう", "。", "」", "つぎ", "？", "）"))
    assert flags == [False, False, True, False, False, True]


def test_sentence_end_flags_without_punctuation():
    assert splitter._sentence_end_flags(words("こんにちは", "みなさん")) == [False, False]


def test_janome_missing_is_reported():
    if splitter._JanomeTokenizer is None:
        assert "janome" in splitter.split_boundary_mode() and "簡易" in splitter.split_boundary_mode()
    else:
        assert splitter.split_boundary_mode() == "janome 斷詞"


# ---- 沒做對齊時，拆出來的字幕不會一閃即逝 ----
def test_ensure_min_display_extends_but_never_overlaps_the_next():
    segs = [{"start": 0, "end": 0.0, "text": "a"}, {"start": 0.1, "end": 0.2, "text": "b"}, {"start": 5, "end": 5.05, "text": "c"}]
    splitter._ensure_min_display(segs)
    assert [(s["start"], s["end"]) for s in segs] == [(0, 0.1), (0.1, 0.1 + config.SPLIT_MIN_DISPLAY_SEC), (5, 5 + config.SPLIT_MIN_DISPLAY_SEC)]
