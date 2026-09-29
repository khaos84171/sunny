"""對齊後的時間處理:把對齊結果寫回片段、產生需要人工檢查的報告、最後的時間微調。"""

from __future__ import annotations

from . import config
from .models import Word
from .srt_io import format_timestamp


def apply_alignment(segments: list[dict], result: dict, log_func):
    """
    把對齊結果寫回 segments（原地修改）：整段的起訖時間，以及每個詞的時間（有 words 的話）。
    對不上的片段保留原本時間。另外記下 orig_start / conf / wide / aligned 給檢查報告用。
    """
    shifts = []
    for seg, span, conf, wide, wspans in zip(segments, result["spans"], result["confs"],
                                             result["wide"], result["word_spans"]):
        seg["orig_start"] = seg["start"]
        seg["conf"] = conf
        seg["wide"] = bool(wide)
        seg["aligned"] = span is not None
        if span is None:
            continue
        shifts.append(abs(span[0] - seg["start"]))
        orig_start, orig_end = seg["start"], seg["end"]
        seg["start"], seg["end"] = span[0], span[1]
        words = seg.get("words")
        if words and wspans and len(wspans) == len(words):
            new_words, prev_end = [], span[0]
            for w, t in zip(words, wspans):
                if t is None:  # 純標點之類不會發音的詞：時間黏在前一個詞後面
                    new_words.append(Word(prev_end, prev_end, w.word))
                else:
                    new_words.append(Word(t[0], t[1], w.word))
                    prev_end = t[1]
            seg["words"] = new_words
        elif words:
            # 沒拿到詞時間（例如還在用舊版 align_worker.py）：把 Whisper 的詞時間
            # 等比例搬進對齊後的範圍，至少整段的起訖會是對齊後的時間，不會被詞時間蓋回去
            scale = (span[1] - span[0]) / (orig_end - orig_start) if orig_end > orig_start else 0.0

            def remap(t: float) -> float:
                return min(max(span[0] + (t - orig_start) * scale, span[0]), span[1])

            seg["words"] = [Word(remap(w.start), remap(w.end), w.word) for w in words]

    avg_shift = sum(shifts) / len(shifts) if shifts else 0.0
    log_func(f"[對齊] 完成：{len(shifts)}/{len(segments)} 條成功對齊，起點平均移動 {avg_shift:.2f} 秒")


def report_alignment(segments: list[dict], subs: list[dict], log_func):
    """
    列出需要人工確認的片段：對不上的、信心偏低的（通常是 Whisper 聽錯字或幻覺）、
    起點被大幅移動的（多半是修正，但值得確認）。編號是輸出 SRT 裡的字幕編號，
    一段被拆成好幾條時顯示成「#12～14」。subs 要傳「最後真正寫進 SRT 的列表」
    （包含開頭空白字幕），編號才會跟 SRT 對得上；沒有 src 的字幕（空白字幕）不列。
    """
    numbers: dict[int, list[int]] = {}
    for no, sub in enumerate(subs, start=1):
        if sub.get("src") is not None:
            numbers.setdefault(sub["src"], []).append(no)

    suspicious = []
    for i, seg in enumerate(segments):
        if not seg["text"].strip():
            continue
        conf = seg.get("conf")
        if not seg.get("aligned"):
            why = "對不上，保留原時間"
        elif conf is not None and conf < config.ALIGN_LOW_CONF:
            why = f"信心 {conf:.2f}"
        elif abs(seg["start"] - seg["orig_start"]) >= config.ALIGN_BIG_SHIFT_SEC:
            why = (f"起點移動 {seg['start'] - seg['orig_start']:+.1f} 秒"
                   + ("，大範圍重對" if seg.get("wide") else ""))
        else:
            continue
        nos = numbers.get(i, [i + 1])
        label = f"#{nos[0]}" if len(nos) == 1 else f"#{nos[0]}～{nos[-1]}"
        suspicious.append(f"    {label} [{format_timestamp(seg['start'])}]（{why}）{seg['text']}")

    if suspicious:
        log_func(f"[對齊] 以下 {len(suspicious)} 處建議檢查（時間可能不準，或字幕文字跟實際說的不一樣）：")
        for line in suspicious[:30]:
            log_func(line)
        if len(suspicious) > 30:
            log_func(f"    …還有 {len(suspicious) - 30} 處，請看 whisper_app.log")
            for line in suspicious[30:]:
                print(line)


def finalize_aligned_timing(subs: list[dict], duration: float):
    """
    對齊後的後處理：稍微提早出現、不重疊、最短顯示時間、結尾稍微延長。
    最短顯示時間與結尾延長都只用到跟下一句之間的空隙：不會蓋到下一句，
    也不會把下一句的起點往後推（空隙不夠就少延長，寧可短一點）。
    先把所有起點提早，再處理重疊與延長，後一句提早的空間才不會被前一句的延長吃掉。
    """
    for sub in subs:
        if sub.get("aligned"):
            sub["start"] = max(0.0, sub["start"] - config.ALIGN_START_LEAD_SEC)
    for i, sub in enumerate(subs):
        if i > 0 and sub["start"] < subs[i - 1]["end"]:
            sub["start"] = subs[i - 1]["end"]
        next_start = subs[i + 1]["start"] if i + 1 < len(subs) else duration
        want = max(sub["end"], sub["start"] + config.ALIGN_MIN_DURATION) + config.ALIGN_END_HOLD_SEC
        sub["end"] = max(sub["end"], min(want, next_start))
        if sub["end"] <= sub["start"]:
            # 前一句蓋過了整句的極端情況：至少給最短顯示時間，後一句的起點下一輪會被推開
            sub["end"] = sub["start"] + config.ALIGN_MIN_DURATION
