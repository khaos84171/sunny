"""字幕拆分:用 word timestamps 把塞了好幾句的片段拆開。有裝 janome 時只在「文節」交界切。"""

from __future__ import annotations

import os
import re
import tempfile
import unicodedata

from . import config


# ---- 哪些「詞與詞之間」可以切 ----
# Whisper 給的日文「詞」其實是 token：「熊本」會變成「熊」「本」兩個。對齊後的停頓又是用
# wav2vec2 的字間空白量的，一個漢字念好幾拍（熊＝くま），同一個詞的兩個漢字之間也可能
# 看起來像停頓。所以先用 janome 斷詞，只允許在「文節」的交界切：
#   ・不在一個詞的中間切（熊｜本）
#   ・助詞、助動詞、接尾詞、標點黏在前面（熊本｜における、8｜年）
#   ・接續詞、連體詞、接頭詞、フィラー黏在後面（そして｜今回の、その｜問題）
#   ・複合名詞不拆（熊本｜地震、早押し｜クイズ），「名詞＋する」不拆（勉強｜する）
try:
    from janome.tokenizer import Tokenizer as _JanomeTokenizer
except ImportError:
    _JanomeTokenizer = None


_janome = None


def _get_janome():
    """第一次用到才建立。hotwords 候選詞加進使用者字典，人名和「早押し」這類詞才不會被拆錯。"""
    global _janome
    if _janome is None and _JanomeTokenizer is not None:
        with tempfile.TemporaryDirectory(prefix="whisper_janome_") as tmp:
            udic = os.path.join(tmp, "hotwords.csv")
            with open(udic, "w", encoding="utf-8") as f:
                for w in config.HOTWORD_NAME_CANDIDATES + config.HOTWORD_COMMON_CANDIDATES:
                    if "," not in w:
                        f.write(f"{w},名詞,{w}\n")
            _janome = _JanomeTokenizer(udic, udic_type="simpledic", udic_enc="utf8")
    return _janome


def split_boundary_mode() -> str:
    return "janome 斷詞" if _JanomeTokenizer is not None else "簡易規則（建議 pip install janome）"


def _can_break(prev: list[str], cur: list[str]) -> bool:
    """兩個 janome 詞（品詞欄位 list）之間是不是文節交界。"""
    if cur[0] in ("助詞", "助動詞"):
        return False
    if cur[0] == "記号" and cur[1] != "括弧開":
        return False
    if cur[1] in ("接尾", "非自立"):
        return False
    if prev[0] in ("接頭詞", "連体詞", "接続詞", "フィラー"):
        return False
    if prev[0] == "記号" and prev[1] == "括弧開":
        return False
    if prev[0] == "名詞" and cur[0] == "名詞" and prev[1] != "副詞可能":
        return False  # 複合名詞（「今日」「今回」這類時間名詞後面可以切）
    if prev[0] == "名詞" and prev[1] == "サ変接続" and cur[0] == "動詞":
        return False
    return True


def _janome_break_offsets(text: str) -> set[int]:
    """回傳 text 裡可以切開的字元位置。"""
    spans = []  # 不含空白的詞：(起點, 終點, 品詞欄位)
    pos = 0
    for t in _get_janome().tokenize(text):
        start, pos = pos, pos + len(t.surface)
        fields = t.part_of_speech.split(",")
        if fields[0] == "記号" and fields[1] == "空白":
            continue
        if t.surface in config.PARTICLE_SURFACES:
            fields = ["助詞", "*"]
        spans.append((start, pos, fields))
    ok = set()
    for (_, e1, p1), (s2, _, p2) in zip(spans, spans[1:]):
        if _can_break(p1, p2):
            ok.update(range(e1, s2 + 1))  # 中間有空白的話，空白前後都算
    return ok


def _char_class(ch: str) -> str:
    if ch == "々" or "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
        return "kanji"
    if "\u30a1" <= ch <= "\u30ff" or "\uff66" <= ch <= "\uff9f":
        return "katakana"
    if unicodedata.normalize("NFKC", ch).isascii() and ch.isalnum():
        return "alnum"
    return "other"


def _heuristic_can_break(left_text: str, right_word: str) -> bool:
    """沒裝 janome 時的替代規則：只看切點兩邊的字。"""
    a, b = left_text.rstrip(), right_word.lstrip()
    if not a or not b:
        return True
    if b[0] in "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶーｰ゛゜" or b[0] in config.CLOSING_CHARS or b in config.PARTICLE_SURFACES:
        return False
    ca, cb = _char_class(a[-1]), _char_class(b[0])
    if ca == cb and ca in ("kanji", "katakana", "alnum") and not right_word[:1].isspace():
        return False  # 熊｜本、ツア｜ー、Quiz｜Knock
    if ca == "alnum" and cb == "kanji":
        return False  # 8｜年
    return not a.endswith(config.ATTACH_NEXT_WORDS)


def _allowed_cuts(words) -> list[bool]:
    """每一對相鄰的詞（words[k] 和 words[k+1]）之間能不能切，長度 = len(words) - 1。"""
    if len(words) < 2:
        return []
    if _JanomeTokenizer is not None:
        ok = _janome_break_offsets("".join(w.word for w in words))
        result, end = [], 0
        for w in words[:-1]:
            end += len(w.word)
            result.append(end in ok)
        return result
    result, left = [], ""
    for w, nxt in zip(words, words[1:]):
        left += w.word
        result.append(_heuristic_can_break(left, nxt.word))
    return result


def _char_count(s: str) -> int:
    """字數（不算空白）。"""
    return len(re.sub(r"\s", "", s))


def _words_text(words) -> str:
    return "".join(w.word for w in words).strip()


def _last_char(word) -> str:
    t = word.word.strip()
    return t[-1] if t else ""


def _sentence_end_flags(words) -> list[bool]:
    """
    每個詞是不是「句子在這裡結束」。句尾標點後面緊跟右括號／引號時，句尾記在最後一個
    括號上（。」→ 切在」之後）；標點跟括號黏在同一個詞裡（。」）也認得。
    """
    flags = []
    for w in words:
        t = w.word.strip()
        core = t.rstrip(config.CLOSING_CHARS)
        if core:
            flags.append(core[-1] in config.SENTENCE_END_CHARS)
        elif t and flags:  # 這個詞只有括號：沿用前一個詞的句尾狀態，切點往後移
            flags[-1], moved = False, flags[-1]
            flags.append(moved)
        else:
            flags.append(False)
    return flags


def _too_long(words) -> bool:
    return (_char_count(_words_text(words)) > config.SPLIT_MAX_CHARS
            or words[-1].end - words[0].start > config.SPLIT_MAX_DURATION)


def _best_cut(words, ok: list[bool]) -> int | None:
    """
    在過長的片段裡找最適合切開的位置，回傳「切在第 k 個詞之後」的 k。
    分數 = 停頓秒數 + （前一個詞以「、」或けど／から等結尾就加分）
          − （下一條會以單獨助詞開頭就扣分）+ （前後長度越平均越加分）。
    兩邊都至少要有 SPLIT_MIN_CHARS 個字，而且只考慮 ok[k] 為 True（文節交界）的位置；
    整段都沒有合法位置（例如一長串複合名詞）才退回任意位置。找不到回傳 None。
    """
    for restrict in (True, False):
        k = _best_cut_pass(words, ok if restrict else None)
        if k is not None:
            return k
    return None


def _best_cut_pass(words, ok: list[bool] | None) -> int | None:
    total = _char_count(_words_text(words))
    best_k, best_score = None, float("-inf")
    left = 0
    for k in range(len(words) - 1):
        left += _char_count(words[k].word)
        right = total - left
        if left < config.SPLIT_MIN_CHARS or right < config.SPLIT_MIN_CHARS:
            continue
        if ok is not None and not ok[k]:
            continue
        score = max(0.0, words[k + 1].start - words[k].end)
        if _last_char(words[k]) in config.SOFT_BREAK_CHARS:
            score += 1.0
        elif words[k].word.strip().endswith(config.CLAUSE_END_SUFFIXES):
            score += 0.6
        elif words[k].word.strip() in config.AFTER_PARTICLES:
            score += 0.3
        if words[k + 1].word.strip() in config.NO_START_PARTICLES:
            score -= 1.0
        score += 0.5 * (1 - abs(left - right) / max(total, 1))
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _split_long(words, ok: list[bool]) -> list:
    """太長就切一刀，兩半各自再檢查，直到都符合上限（或已經切不動）。ok 長度 = len(words) - 1。"""
    if len(words) < 2 or not _too_long(words):
        return [words]
    k = _best_cut(words, ok)
    if k is None:
        return [words]
    return _split_long(words[:k + 1], ok[:k]) + _split_long(words[k + 1:], ok[k + 1:])


def split_segment(seg: dict) -> list[dict]:
    """
    把一個片段 {"start", "end", "text", "words": list[Word] 或 None} 拆成多條字幕，
    回傳 [{"start", "end", "text"}, ...]。沒有詞時間時原樣回傳一條。
    words 的時間可能是 Whisper 估的，也可能是對齊後 wav2vec2 的（apply_alignment 換過）。
    """
    words = [w for w in (seg.get("words") or []) if w.word.strip()]
    if not words:
        return [{"start": seg["start"], "end": seg["end"], "text": seg["text"]}]

    ok = _allowed_cuts(words)  # 哪些位置是文節交界（不會切在一個詞的中間）
    ends = _sentence_end_flags(words)  # 哪些詞是句尾（已把後面的右括號算進去）

    # 第 1、2 步：句尾標點 → 切；明顯停頓而且剛好在文節交界 → 切
    groups, start = [], 0
    for i in range(len(words) - 1):
        w, nxt = words[i], words[i + 1]
        if ends[i]:
            cut = True
        else:
            cut = (ok[i] and nxt.start - w.end >= config.SPLIT_PAUSE_SEC
                   and _char_count(_words_text(words[start:i + 1])) >= config.SPLIT_MIN_CHARS)
        if cut:
            groups.append((start, i + 1))
            start = i + 1
    groups.append((start, len(words)))

    # 第 3 步：還是太長的再切
    pieces = []
    for a, b in groups:
        for part in _split_long(words[a:b], ok[a:b - 1]):
            pieces.append({"start": part[0].start, "end": part[-1].end, "text": _words_text(part)})
    return pieces


def _ensure_min_display(segments: list[dict]):
    """沒做對齊時，避免拆出來的字幕長度是 0 或一閃即逝（不會蓋到下一條）。"""
    for i, seg in enumerate(segments):
        if seg["end"] - seg["start"] >= config.SPLIT_MIN_DISPLAY_SEC:
            continue
        limit = segments[i + 1]["start"] if i + 1 < len(segments) else float("inf")
        seg["end"] = max(seg["end"], min(seg["start"] + config.SPLIT_MIN_DISPLAY_SEC, limit))
