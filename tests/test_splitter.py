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


# ---- 接在前一句後面的詞：標點或停頓後面緊跟它們時不切（字幕實例：#231、#239、#242、#676） ----
@pytest.mark.parametrize("tokens", [
    ("どの", "方角", "でしょう", "?", "だ", "と", "思う", "んだ", "けど", "。"),                              # ?だと思うんだけど。
    ("Bの", "生みの", "親", "は", "誰", "?", "みたい", "に", "なる", "はず", "だ", "。"),                   # ?みたいになるはずだ。
    ("だから", "、", "何々", "は", "誰", "?", "って", "いう", "大元", "の", "日本語", "が", "あって", "、"),  # ?っていう大元の…
    ("Bの", "人物", "は", "誰", "?", "とか", "、"),                                                         # ?とか、
    ("そう", "です", "。", "ね", "。"),                                                                      # 。ね。
])
def test_words_that_belong_to_the_previous_sentence_are_not_cut_off(mode, tokens):
    assert pieces(words(*tokens)) == ["".join(tokens)]


def test_sentence_after_attach_word_is_still_split(mode):
    assert pieces(words("そう", "です", "。", "ね", "。", "次", "行き", "ましょう", "。")) == ["そうです。ね。", "次行きましょう。"]


def test_attach_word_is_not_cut_off_by_a_pause_either(mode):
    assert pieces(words("難しい", "です", "けど", "面白い", "です", "。", gaps={1: 0.6})) == ["難しいですけど面白いです。"]


def test_new_sentences_that_look_like_particles_still_start_a_subtitle(mode):
    assert pieces(words("ありがとう", "。", "よし", "、", "行こう", "。")) == ["ありがとう。", "よし、行こう。"]
    assert pieces(words("お願い", "します", "!", "という", "こと", "で", "、", "次", "です", "。")) == ["お願いします!", "ということで、次です。"]
    assert pieces(words("入り", "まし", "た", "。", "より", "アップデート", "して", "いける", "よう", "に", "頑張り", "ます", "。")) \
        == ["入りました。", "よりアップデートしていけるように頑張ります。"]


@pytest.mark.parametrize("text, expected", [
    ("とか、", True), ("みたいになるはずだ。", True), ("だと思うんだけど。", True), ("っていう大元の", True),
    ("とかは聞くけど、", True), ("ぐらいにしないとでも文字数的には入らないと思う。", True), ("のだと、", True),
    ("みたいなね。", True), ("になってるわけだから、", True), ("ね。", True),
    ("よし!", False), ("よいしょ!", False), ("ということで、", False), ("となると、", False),
    ("よりアップデートしていけるように", False), ("ねえ牛とらう、", False), ("よしよし。", False),
])
def test_attach_prefix_on_real_subtitle_texts(mode, text, expected):
    assert (splitter._attach_prefix_len(text) > 0) is expected


# ---- 停頓切：句子還沒說完（停在助詞）、碎片太短都不切（字幕實例：#117、#283、#387） ----
def test_short_head_before_a_particle_is_not_cut_by_a_pause(mode):
    assert pieces(words("私", "は", "昨日", "東大", "に", "行き", "まし", "た", "。", gaps={1: 0.6})) == ["私は昨日東大に行きました。"]


def test_pause_after_a_particle_needs_a_longer_pause(mode):
    tokens = ("では", "、", "みなさん", "で", "クイズノック", "記録", "を", "出し", "ましょう", "。")
    assert pieces(words(*tokens, gaps={6: 0.6})) == ["では、みなさんでクイズノック記録を出しましょう。"]
    assert pieces(words(*tokens, gaps={6: 1.2})) == ["では、みなさんでクイズノック記録を", "出しましょう。"]


def test_pause_does_not_leave_a_tiny_tail(mode):
    tokens = ("尾崎", "幸男", "は", "20", "何回", "と", "か", "で", "、", "衆議院", "議員", "を", "6", "度", "。")
    assert pieces(words(*tokens, gaps={11: 0.5})) == ["尾崎幸男は20何回とかで、", "衆議院議員を6度。"]   # 在「、」切；「を｜6度。」不切


def test_pause_after_a_comma_may_still_cut_a_short_head(mode):
    assert pieces(words("はい", "、", "こちら", "は", "人気", "企画", "です", gaps={1: 0.6})) == ["はい、", "こちらは人気企画です"]


def test_a_finished_greeting_ending_in_ha_is_still_cut_by_a_pause(mode):
    assert pieces(words("こんにちは", "みなさん", gaps={0: 1.0})) == ["こんにちは", "みなさん"]


# ---- 長度切：多一兩個字不硬切；修飾語「の」和名詞不拆（字幕實例：#160/161、#522/523） ----
def test_a_couple_of_chars_over_the_limit_is_left_alone(mode):
    limit = config.SPLIT_MAX_CHARS + config.SPLIT_OVERFLOW_CHARS
    assert len(pieces(words(*["あいう"] * (limit // 3)))) == 1
    assert len(pieces(words(*["あいう"] * (limit // 3 + 1)))) == 2


def test_long_cut_does_not_separate_no_from_its_noun(mode):
    tokens = ("じゃあ", "そしたら", "ミッキー", "マウス", "の", "生み", "の", "親", "でも", "ある", "みたい", "に", "落とせる", "と", "。")
    result = pieces(words(*tokens))
    assert len(result) == 2 and not any(p.startswith("親") for p in result)


# ---- 跨 Whisper 片段的合併（字幕實例：#130/131「はい／ー。」、#232、#283） ----
def sub(start, end, text, src, aligned=True):
    return {"start": start, "end": end, "text": text, "src": src, "aligned": aligned}


def test_merge_note_names_the_reason(mode):
    _, notes = splitter.merge_fragments([sub(0.0, 1.0, "だから、何々は誰?", 0), sub(1.1, 2.0, "っていう大元の", 1)])
    assert "接在前一句後面的詞" in notes[0]


def test_orphan_long_vowel_is_merged_into_the_previous_subtitle():
    merged, notes = splitter.merge_fragments([sub(0.0, 0.6, "はい", 0), sub(0.7, 1.3, "ー。", 1)])
    assert [(m["start"], m["end"], m["text"], m["srcs"]) for m in merged] == [(0.0, 1.3, "はいー。", [0, 1])]
    assert len(notes) == 1 and "はい" in notes[0] and "ー。" in notes[0]


@pytest.mark.parametrize("a, b", [
    ("難しいです。", "けど面白いです。"),                       # 後一條是接在前一句後面的詞
    ("でもペアの強みって", "押した後だと思うんですよ。"),        # 前一條停在助詞上
    ("皆さんも", "クリアしたいですよね。"),
])
def test_split_sentence_across_two_whisper_segments_is_merged(mode, a, b):
    merged, _ = splitter.merge_fragments([sub(0.0, 1.0, a, 0), sub(1.2, 2.0, b, 1)])
    assert [m["text"] for m in merged] == [a + b]


@pytest.mark.parametrize("a, b, gap", [
    ("お勤めた人物は", "ミッキー。", 0.76),                       # 空隙太大：多半是換人講話
    ("こんにちは", "みなさん", 0.1),                              # 前一條說完了、後一條也不是接續詞
    ("ありがとう。", "次の文です。", 0.1),
])
def test_unrelated_neighbours_are_not_merged(mode, a, b, gap):
    merged, notes = splitter.merge_fragments([sub(0.0, 1.0, a, 0), sub(1.0 + gap, 2.0, b, 1)])
    assert [m["text"] for m in merged] == [a, b] and notes == []


def test_pieces_of_the_same_whisper_segment_are_never_merged(mode):
    merged, _ = splitter.merge_fragments([sub(0.0, 1.0, "難しいです。", 3), sub(1.0, 2.0, "けど面白いです。", 3)])
    assert len(merged) == 2


def test_merge_respects_the_length_limit(mode):
    a = "あ" * 20 + "は"
    short, _ = splitter.merge_fragments([sub(0.0, 1.0, a, 0), sub(1.1, 2.0, "い" * 5 + "。", 1)])
    assert len(short) == 1                                   # 同樣的接續關係，沒超過上限就會併
    long, _ = splitter.merge_fragments([sub(0.0, 1.0, a, 0), sub(1.1, 2.0, "い" * 12 + "。", 1)])
    assert len(long) == 2                                    # 併起來超過字數上限就不併


def test_join_text_adds_a_space_only_between_latin_words():
    assert splitter._join_text("Quiz", "Knock") == "Quiz Knock"
    assert splitter._join_text("はい", "ー。") == "はいー。"
    assert splitter._join_text("GameKnack", "の") == "GameKnackの"


def test_merging_a_chain_tracks_every_source_segment():
    merged, _ = splitter.merge_fragments([sub(0.0, 0.5, "はい", 0), sub(0.6, 1.0, "ー", 1), sub(1.1, 1.5, "ー。", 2), sub(5.0, 6.0, "次です。", 3)])
    assert [(m["text"], m["srcs"], m["end"]) for m in merged[:1]] == [("はいーー。", [0, 1, 2], 1.5)]
    assert merged[1]["text"] == "次です。" and "srcs" not in merged[1]
