"""
多模型交叉比對:Whisper 跟 Qwen3-ASR、Parakeet 的辨識結果逐段比較，多數決修正 Whisper 聽錯的字（ROVER）。

做法（每一條字幕各自做）：
    1. 把三個結果轉成「比較用的字串」：去掉標點與空白、全形半形統一、片假名當平假名、漢數字當阿拉伯數字，
       記下每個字在原文的位置。
    2. 用編輯距離把另外兩個結果逐字對到 Whisper 上。三個都對得上同一個字的地方是「錨點」。
    3. 錨點之間的每一段各自投票：另外兩個模型在這段寫得一樣、而且跟 Whisper 不同，就換成它們的寫法（2 票勝 1 票）。
       換的時候只換這一段，Whisper 的標點、前後文、詞的時間都保留。
    4. 下列情況有多數也不改（只記下來）：
       ・字幕開頭／結尾「多一段」或「少一段」：切出來的聲音常常多包到、少包到前後句的一點點
       ・差別只是多了／少了語助詞（えー、あの…）
       ・一邊是漢字、一邊只有假名（「一人」／「ひとり」）：多半只是寫法不同
       ・讀音一樣、只是寫法不同（有裝 janome 時才判斷得到，例如「伊沢」／「井沢」、「勝った」／「買った」）
       ・這次勾選的 hotwords（人名）：Whisper 有拿到提示，以 Whisper 為準
       ・一次要換太多字（CROSS_MAX_CHANGE_CHARS）：一大段完全不同，多半是聲音切錯或整句幻覺，留給人工判斷
主程式這一側的模型呼叫在 cross_asr.py；這裡只有純文字的比較，不需要模型也能測試。
"""

from __future__ import annotations

import collections
import re
import unicodedata

from . import config
from .models import Word
from .srt_io import format_timestamp

try:
    from janome.tokenizer import Tokenizer as _JanomeTokenizer
except ImportError:
    _JanomeTokenizer = None


_KANJI_DIGITS = str.maketrans("〇一二三四五六七八九", "0123456789")


def _to_hiragana(ch: str) -> str:
    code = ord(ch)
    return chr(code - 0x60) if 0x30A1 <= code <= 0x30F6 else ch


def key_chars(text: str) -> tuple[list[str], list[int]]:
    """比較用的字（只留文字與數字）以及每個字在 text 裡的位置。"""
    keys, pos = [], []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch).lower():
            c = _to_hiragana(c).translate(_KANJI_DIGITS)
            if unicodedata.category(c)[0] in "LN":
                keys.append(c)
                pos.append(i)
    return keys, pos


def normalize_key(text: str) -> str:
    return "".join(key_chars(text)[0])


def match_positions(a: list[str], b: list[str]) -> tuple[list[int | None], int]:
    """
    編輯距離逐字對齊。回傳 (m, 距離)：m[i] = a[i] 對到的 b 的位置（同一個字才算），對不到是 None。
    同分時優先「對到同一個字」，讓錨點越多越好。
    """
    n, k = len(a), len(b)
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(k + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        row, prev, ai = dp[i], dp[i - 1], a[i - 1]
        for j in range(1, k + 1):
            row[j] = min(prev[j - 1] + (ai != b[j - 1]), prev[j] + 1, row[j - 1] + 1)
    m: list[int | None] = [None] * n
    i, j = n, k
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            m[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return m, dp[n][k]


def similarity(a: str, b: str) -> float:
    """兩個結果有多像（比較用的字串，0～1）。"""
    ka, kb = normalize_key(a), normalize_key(b)
    if not ka and not kb:
        return 1.0
    return 1.0 - match_positions(list(ka), list(kb))[1] / max(len(ka), len(kb))


_filler_re = None


def _only_fillers(key: str) -> bool:
    global _filler_re
    if _filler_re is None:
        alts = sorted({normalize_key(f) for f in config.CROSS_FILLERS if normalize_key(f)}, key=len, reverse=True)
        _filler_re = re.compile("(?:" + "|".join(map(re.escape, alts)) + ")+") if alts else re.compile("(?!)")
    return bool(key) and _filler_re.fullmatch(key) is not None


_janome = None


def _reading(text: str) -> str | None:
    """讀音（平假名，比較用）；沒裝 janome 時回傳 None。"""
    global _janome
    if _JanomeTokenizer is None:
        return None
    if _janome is None:
        _janome = _JanomeTokenizer()
    parts = []
    for t in _janome.tokenize(text):
        parts.append(t.surface if t.reading in ("*", "") else t.reading)
    return normalize_key("".join(parts))


def _is_kana(text: str) -> bool:
    """只有平假名／片假名（和長音、標點），沒有漢字、英數字。"""
    keys = key_chars(text)[0]
    return bool(keys) and all("\u3041" <= k <= "\u3096" or k in "ー゛゜" for k in keys)


def _same_reading(primary: str, start: int, end: int, new_text: str) -> bool:
    """換掉之後整句的讀音跟原本一樣（伊沢／井沢）：只是寫法不同，不是聽錯。整句一起讀，漢字的讀法才準。"""
    before = _reading(primary)
    if before is None:
        return False
    return (before == _reading(primary[:start] + new_text + primary[end:])
            or _reading(primary[start:end]) == _reading(new_text))  # 人名之類整句讀錯時，單獨讀常常是對的


Edit = collections.namedtuple("Edit", "start end old new voters")        # 把 text[start:end] 換成 new
Held = collections.namedtuple("Held", "old new voters reason")            # 有多數但沒有改的地方


def _protected_spans(primary: str, protect) -> list[tuple[int, int]]:
    spans = []
    for word in protect or ():
        if word:
            spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(word), primary))
    return spans


def vote(primary: str, others: dict[str, str | None], protect=()) -> tuple[list[Edit], list[Held]]:
    """
    primary（Whisper）跟 others（{模型名稱: 結果，None = 這個模型沒有結果}）逐段投票。
    protect：不能改的詞（這次勾選的 hotwords：Whisper 有拿到提示，其他模型不一定有，人名以 Whisper 為準）。
    回傳 (要做的修改, 有多數但沒改的差異)；修改的位置是 primary 裡的字元位置，照先後排好、不重疊。
    """
    guarded = _protected_spans(primary, protect)
    hyps = {name: text for name, text in others.items() if text is not None}
    if len(hyps) < max(config.CROSS_MIN_VOTES, 1):
        return [], []
    a_keys, a_pos = key_chars(primary)
    if not a_keys:
        return [], []
    h_info = {}
    for name, text in hyps.items():
        keys, pos = key_chars(text)
        h_info[name] = (text, keys, pos, match_positions(a_keys, keys)[0])

    n = len(a_keys)
    anchors = [i for i in range(n) if all(info[3][i] is not None for info in h_info.values())]
    edits, held = [], []
    prev = -1
    for nxt in anchors + [n]:
        a_key = "".join(a_keys[prev + 1:nxt])
        regions = {}
        for name, (text, keys, pos, m) in h_info.items():
            lo = m[prev] + 1 if prev >= 0 else 0
            hi = m[nxt] if nxt < n else len(keys)
            h_key = "".join(keys[lo:hi])
            h_text = text[pos[lo]:pos[hi - 1] + 1] if hi > lo else ""
            regions[name] = (h_key, h_text, lo, hi, pos)
        counts = collections.Counter(r[0] for r in regions.values())
        counts[a_key] += 1  # Whisper 自己也是一票
        winner, votes = max(((k, c) for k, c in counts.items() if k != a_key), key=lambda kc: kc[1], default=(None, 0))
        if winner is not None and votes >= config.CROSS_MIN_VOTES and votes > counts[a_key]:
            voters = [name for name, r in regions.items() if r[0] == winner]
            new_text = regions[voters[0]][1]
            if a_key:
                start, end = a_pos[prev + 1], a_pos[nxt - 1] + 1
            else:  # 插入：放在前一個字後面（標點前面）
                start = end = a_pos[prev] + 1 if prev >= 0 else a_pos[0]
            old_text = primary[start:end]
            at_edge = prev < 0 or nxt >= n
            reason = None
            if at_edge and (not a_key or not winner):
                reason = "edge"
            elif any(start < g_end and g_start < max(end, start + 1) for g_start, g_end in guarded):
                reason = "hotword"
            elif _only_fillers(a_key or winner) and (not a_key or not winner):
                reason = "filler"
            elif max(len(a_key), len(winner)) > config.CROSS_MAX_CHANGE_CHARS:
                reason = "too_long"
            elif not _clean_cut(a_pos, prev, nxt) or not all(_clean_cut(regions[v][4], regions[v][2] - 1, regions[v][3])
                                                              for v in voters):
                reason = "split_char"   # 一個字（例如「㍻」）展開成好幾個比較用的字，差異切在它中間：不動
            elif a_key and winner and _is_kana(old_text) != _is_kana(new_text):
                reason = "spelling"     # 一邊是漢字、一邊只有假名（一人／ひとり）：多半只是寫法不同
            elif a_key and winner and _same_reading(primary, start, end, new_text):
                reason = "same_reading"
            if reason is None:
                edits.append(Edit(start, end, old_text, new_text, voters))
            else:
                held.append(Held(old_text, new_text, voters, reason))
        prev = nxt
    return edits, held


def _clean_cut(pos: list[int], prev: int, nxt: int) -> bool:
    """[prev+1, nxt) 這段的邊界沒有把同一個原文字元拆開。"""
    if 0 <= prev < len(pos) - 1 and pos[prev] == pos[prev + 1]:
        return False
    if 0 < nxt < len(pos) and pos[nxt - 1] == pos[nxt]:
        return False
    return True


def apply_edits(text: str, edits: list[Edit]) -> str:
    for e in reversed(edits):
        text = text[:e.start] + e.new + text[e.end:]
    return text


def apply_edits_to_words(words: list, edits: list[Edit]) -> list:
    """
    把修改套用到詞的列表上（words 接起來就是修改用的原文）。每個字記得自己屬於哪個詞：
    換進來的字歸到被換掉的第一個字所屬的詞（插入時歸到前一個字的詞），詞的時間不變；整個被刪光的詞拿掉。
    """
    chars, owner = [], []
    for wi, w in enumerate(words):
        chars.extend(w.word)
        owner.extend([wi] * len(w.word))
    for e in reversed(edits):
        if e.start < e.end:
            who = owner[e.start]
        elif e.start > 0:
            who = owner[e.start - 1]
        else:
            who = owner[0] if owner else 0
        chars[e.start:e.end] = list(e.new)
        owner[e.start:e.end] = [who] * len(e.new)
    out, cur, cur_owner = [], [], None
    for ch, who in zip(chars, owner):
        if who != cur_owner and cur:
            out.append(Word(words[cur_owner].start, words[cur_owner].end, "".join(cur)))
            cur = []
        cur_owner = who
        cur.append(ch)
    if cur:
        out.append(Word(words[cur_owner].start, words[cur_owner].end, "".join(cur)))
    return [w for w in out if w.word.strip()] or out[:1]


def apply_crosscheck(segments: list[dict], hyps: list[dict], labels: dict[str, str], log_func, protect=()) -> dict:
    """
    segments：Whisper 的片段（原地修改文字與詞）；hyps：每個片段 {模型名稱: 結果 或 None}；labels：模型名稱 → 顯示用名稱；
    protect：不能改的詞（hotwords）。
    回傳統計 {"fixed", "edits", "same", "review", "unheard", "dropped"}。
    """
    stats = collections.Counter()
    review, fixes = [], []
    keep = []
    for seg, others in zip(segments, hyps):
        available = {k: v for k, v in others.items() if v is not None}
        seg["cross"] = dict(available)
        text = seg["text"]
        a_key = normalize_key(text)
        when = format_timestamp(seg["start"])
        if not a_key or not available:
            keep.append(seg)
            continue
        if all(normalize_key(v) == a_key for v in available.values()):
            stats["same"] += 1
        heard = [k for k, v in available.items() if normalize_key(v)]
        if not heard and len(available) >= config.CROSS_MIN_VOTES:
            stats["unheard"] += 1
            if config.CROSS_DROP_UNHEARD:
                stats["dropped"] += 1
                log_func(f"[交叉比對] 刪除 [{when}] {text}（其他模型都聽不到任何字）")
                continue
            review.append(f"    [{when}]（其他模型都聽不到任何字，可能是幻覺）{text}")
            keep.append(seg)
            continue

        words = seg.get("words")
        surface = "".join(w.word for w in words) if words else text
        edits, held = vote(surface, available, protect)
        if edits:
            new_surface = apply_edits(surface, edits)
            if words:
                seg["words"] = apply_edits_to_words(words, edits)
                seg["text"] = new_surface.strip()
            else:
                seg["text"] = new_surface
            seg["cross_fixed"] = True
            stats["fixed"] += 1
            stats["edits"] += len(edits)
            changes = "、".join(f"「{e.old}」→「{e.new}」" for e in edits)
            who = "、".join(labels.get(v, v) for v in edits[0].voters)
            fixes.append(f"    [{when}] {changes}（{who} 一致）\n        {text}\n      → {seg['text']}")
        elif max((similarity(text, v) for v in available.values()), default=1.0) < config.CROSS_REVIEW_SIM:
            shown = "／".join(f"{labels.get(k, k)}：{v or '（沒聽到）'}" for k, v in available.items())
            review.append(f"    [{when}]（跟其他模型差很多）{text}\n        {shown}")
        for h in held:
            if h.reason == "too_long":
                review.append(f"    [{when}]（其他模型一致聽成「{h.new}」，差太多沒有自動改）{text}")
        keep.append(seg)
    segments[:] = keep

    log_func(f"[交叉比對] 完成：{len(hyps)} 條中 {stats['same']} 條各模型完全一致，"
             f"{stats['fixed']} 條依多數修正（共 {stats['edits']} 處）"
             + (f"，{stats['dropped']} 條刪除" if stats["dropped"] else ""))
    for line in fixes:
        log_func(line)
    if review:
        log_func(f"[交叉比對] 以下 {len(review)} 處建議檢查（字幕文字可能跟實際說的不一樣）：")
        for line in review[:30]:
            log_func(line)
        if len(review) > 30:
            log_func(f"    …還有 {len(review) - 30} 處，請看 whisper_app.log")
            for line in review[30:]:
                print(line)
    stats["review"] = len(review)
    return dict(stats)
