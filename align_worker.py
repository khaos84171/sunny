"""
時間戳對齊的獨立工作程序（由 w1_1.py 自動呼叫，不需要手動執行，請跟 w1_1.py 放在同一個資料夾）。

為什麼要獨立成另一個程序：
    faster-whisper（CTranslate2）和 PyTorch 各自帶了一份 cuDNN 9，但小版本不同。
    兩份同時載入同一個程序時會互相拿錯 DLL，出現
    「Could not load symbol cudnnGetLibConfig. Error code 127」。
    分開在不同程序裡跑就不會衝突（跟 Demucs 用獨立程序跑是同樣的道理）。
    這個檔案只用 PyTorch + transformers，完全不載入 faster-whisper。

對齊方式（兩輪）：
    第一輪（逐句、窄範圍）：每句只在 Whisper 給的時間前後 0.4 秒內對齊。
        Whisper 時間大致正確時這樣最穩：就算 Whisper 多聽或聽錯一兩個字，
        也不會被拉去對到遠處無關的聲音上。
    第二輪（大範圍，只處理有問題的句子）：第一輪出現下列狀況的句子才重對——
        ・對齊結果貼在搜尋範圍的邊緣（代表真正的聲音在範圍外，Whisper 時間偏太多）
        ・完全對不上
        ・信心偏低（字被硬塞到不像的聲音上）
        重對時把連續幾句有問題的接在一起、往前後各多找 3 秒，但範圍會被夾在
        前後「第一輪就對得很好」的句子之間，不會越界搶到別句的聲音。
        第二輪結果信心不比第一輪差、而且沒有「句子被拉長」才採用。
        （拉長＝同一句裡相鄰兩個字之間空了超過 1 秒，通常是 Whisper 多聽或聽錯一個字，
          那個字跑去對到遠處無關的聲音。這種結果不採用，保留第一輪。）
    信心＝每個字對到的聲音機率的平均（算術平均）：只錯一兩個字不會拉低太多，
    整句對錯地方才會很低。

    另外：字表裡沒有的字（罕見漢字、英文、數字）用「任何發音都可以」的萬用字元佔位；
    片假名與平假名發音相同，字表只有其中一種時互相替代。

每個詞的時間（給 w1_1.py 的「自動拆分」用）：
    w1_1.py 會把 Whisper 原本的片段（還沒拆）連同「詞的切法」一起送來。這裡照原樣
    整段對齊（文字越長對得越穩），再從每個字的對齊位置算出每個詞的起訖時間傳回去。
    w1_1.py 用這些時間判斷哪裡有停頓、該在哪裡拆——比 Whisper 自己估的詞時間準很多。
    注意 CTC 的特性：每個字只會在開始發音的那一兩個 frame 出現，所以一個詞的「終點」
    是最後一個字開始發音的位置，字拉長音的部分會算在跟下一個詞之間的空白裡。

用法：python align_worker.py job.json result.json
    job.json    {"audio_npy", "segments": [{"start","end","text","words"(可省略)}, ...],
                 "model_name", "device", "sample_rate", "params": {...}}
                 words = 這句拆成詞的文字 list，接起來要等於 text；沒給就當整句是一個詞
    result.json {"spans": [[起點秒, 終點秒] 或 null, ...],
                 "confs": [0～1 或 null, ...], "wide": [這句是否用第二輪結果, ...],
                 "word_spans": [[[詞起點, 詞終點] 或 null（詞裡沒有會發音的字）, ...] 或 null, ...]}
stdout 每行一則訊息給主程式：
    "LOG <文字>"、"PROGRESS <已完成> <總數>"、"ERROR <文字>"
"""

import json
import sys
import traceback
import unicodedata

import numpy as np

STAR_PENALTY = 1.0      # 萬用字元的 log 機率扣分，讓真正認得的字優先
LOOKAHEAD_TOKENS = 10   # 第二輪分批時，預看下一句開頭幾個字當擋板


def emit(kind: str, msg) -> None:
    print(f"{kind} {msg}", flush=True)


# =========================================================
# === 文字 → token ===
# =========================================================
def _kana_swap(ch: str) -> str:
    """片假名 ⇄ 平假名（ァ～ヶ ⇄ ぁ～ゖ，Unicode 剛好差 0x60）。"""
    code = ord(ch)
    if 0x30A1 <= code <= 0x30F6:
        return chr(code - 0x60)
    if 0x3041 <= code <= 0x3096:
        return chr(code + 0x60)
    return ch


def text_to_tokens(text: str, vocab: dict, delim_id, star_id: int, unknown_chars: set) -> list[int]:
    """
    字表裡有的字 → 對應 token；字表裡沒有、但會唸出來的字（字母、數字、漢字）→ 萬用字元；
    標點、符號、空白 → 略過（不會發音）。英文字母大約每 2 個算 1 個萬用字元。
    """
    tokens = []
    latin_run = 0
    for ch in unicodedata.normalize("NFKC", text).lower():
        if ch.isspace() or ch == "|":
            latin_run = 0
            continue
        tid = vocab.get(ch)
        if tid is None:
            tid = vocab.get(_kana_swap(ch))
        if tid is not None and tid != delim_id:
            tokens.append(tid)
            latin_run = 0
            continue
        if unicodedata.category(ch)[0] in "LN" and ch != "ー":
            unknown_chars.add(ch)
            if ch.isascii():
                if latin_run % 2 == 0:
                    tokens.append(star_id)
                latin_run += 1
            else:
                tokens.append(star_id)
                latin_run = 0
        else:
            latin_run = 0
    return tokens


# =========================================================
# === CTC Viterbi 強制對齊 ===
# =========================================================
def ctc_align(log_probs: np.ndarray, tokens: list[int], blank_id: int):
    """
    標準 CTC Viterbi。log_probs: (T, V)，tokens: 要對齊的字元序列。
    回傳每個 token 的 (起始 frame, 結束 frame, 最高 log 機率)；對不上回傳 None。
    """
    T = log_probs.shape[0]
    L = len(tokens)
    repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    if L == 0 or T < L + repeats:
        return None

    ext = np.full(2 * L + 1, blank_id, dtype=np.int64)
    ext[1::2] = tokens
    S = len(ext)
    emit_lp = log_probs[:, ext]  # (T, S)

    allow_skip = np.zeros(S, dtype=bool)
    allow_skip[2:] = (ext[2:] != blank_id) & (ext[2:] != ext[:-2])

    NEG = -np.inf
    dp = np.full(S, NEG, dtype=np.float64)
    dp[0] = emit_lp[0, 0]
    dp[1] = emit_lp[0, 1]
    backptr = np.zeros((T, S), dtype=np.int8)
    idx = np.arange(S)

    for t in range(1, T):
        step = np.concatenate(([NEG], dp[:-1]))
        skip = np.where(allow_skip, np.concatenate(([NEG, NEG], dp[:-2])), NEG)
        cand = np.stack([dp, step, skip])
        choice = cand.argmax(axis=0)
        backptr[t] = choice
        dp = cand[choice, idx] + emit_lp[t]

    s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    if not np.isfinite(dp[s]):
        return None

    states = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        states[t] = s
        # 轉成 Python int：NumPy 2 下 int 減 int8 的結果會被當成 int8，s > 127 就溢位
        s -= int(backptr[t, s])

    is_tok = states % 2 == 1
    frames = np.nonzero(is_tok)[0]
    tok_idx = states[is_tok] // 2
    first = np.full(L, T, dtype=np.int64)
    np.minimum.at(first, tok_idx, frames)
    last = np.full(L, -1, dtype=np.int64)
    np.maximum.at(last, tok_idx, frames)
    best = np.full(L, NEG)
    np.maximum.at(best, tok_idx, emit_lp[frames, states[is_tok]])
    return first, last + 1, best


# =========================================================
# === 主流程 ===
# =========================================================
def main():
    job_path, result_path = sys.argv[1], sys.argv[2]
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)

    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    model_name = job["model_name"]
    device = job["device"]
    sr = int(job["sample_rate"])
    segments = job["segments"]
    p = job["params"]
    pad_sec = float(p["pad_sec"])        # 第一輪：Whisper 時間前後各多取幾秒
    edge_sec = float(p["edge_sec"])      # 對齊結果離搜尋範圍邊緣小於這個值 → 視為被範圍卡住
    wide_sec = float(p["wide_sec"])      # 第二輪：往前後各多找幾秒
    batch_sec = float(p["batch_sec"])    # 第二輪：連續有問題的句子每批最長幾秒
    low_conf = float(p["low_conf"])
    max_inner_gap = float(p["max_inner_gap"])  # 同一句相鄰兩字最多空幾秒，超過視為被拉長

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        emit("LOG", "偵測不到可用的 CUDA GPU（或 PyTorch 裝成 CPU 版），改用 CPU 對齊，速度會慢很多")
        device = "cpu"

    emit("LOG", f"載入對齊模型：{model_name} …（第一次會下載，約 1.2 GB）")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()
    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id  # wav2vec2 CTC 的 blank 就是 <pad>
    delim_id = vocab.get("|")
    star_id = max(vocab.values()) + 1            # 萬用字元：接在字表最後面的額外一欄
    non_blank = np.array([v for v in sorted(set(vocab.values())) if v != blank_id])

    audio = np.load(job["audio_npy"], mmap_mode="r")
    duration = len(audio) / sr
    n = len(segments)

    unknown_chars: set = set()
    seg_tokens = []        # 每句的 token（整句接起來）
    seg_word_ranges = []   # 每句裡每個詞佔 seg_tokens[k] 的哪一段 [a, b)
    for s in segments:
        toks, ranges = [], []
        for w in (s.get("words") or [s["text"]]):
            wt = text_to_tokens(w, vocab, delim_id, star_id, unknown_chars)
            ranges.append((len(toks), len(toks) + len(wt)))
            toks.extend(wt)
        seg_tokens.append(toks)
        seg_word_ranges.append(ranges)
    emit("LOG", f"對齊模型載入完成（字表 {len(vocab)} 個字元），開始對齊 {n} 條字幕")
    if unknown_chars:
        shown = "".join(sorted(unknown_chars)[:40])
        more = f" 等 {len(unknown_chars)} 個" if len(unknown_chars) > 40 else ""
        emit("LOG", f"字表裡沒有、改用萬用字元對齊的字：{shown}{more}")

    def log_probs_for(win_start: float, win_end: float):
        """取一段音訊跑 wav2vec2，回傳 (log 機率含萬用字元欄, 每 frame 秒數)。"""
        chunk = np.asarray(audio[int(win_start * sr): int(win_end * sr)], dtype=np.float32)
        if len(chunk) < int(0.1 * sr):
            return None, 0.0
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
        with torch.inference_mode():
            logits = model(inputs.input_values.to(device)).logits[0]
        lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        if lp.shape[1] <= star_id:  # 模型輸出欄數跟字表不一致時補齊
            lp = np.pad(lp, ((0, 0), (0, star_id - lp.shape[1])), constant_values=-np.inf)
        star = lp[:, non_blank[non_blank < lp.shape[1]]].max(axis=1, keepdims=True) - STAR_PENALTY
        lp = np.concatenate([lp[:, :star_id], star], axis=1)
        return lp, (len(chunk) / sr) / lp.shape[0]

    def run_align(seg_ids, ws, we, extra_tokens=()):
        """
        把 seg_ids 這幾句的文字接起來，對齊到 [ws, we] 這段音訊。
        回傳 {句子編號: {"span", "conf", "head", "tail", "gap", "tok_times"}}；整段對不上回傳 None。
        head / tail = 這句的第一個字／最後一個字離搜尋範圍左／右邊緣幾秒。
        gap = 這句裡相鄰兩個字之間最大的空白秒數。
        tok_times = 這句每個字的 (起點秒, 終點秒)，用來算每個詞的時間。
        """
        if we - ws < 0.1:
            return None
        tokens, offsets = [], []
        for k in seg_ids:
            offsets.append((len(tokens), len(tokens) + len(seg_tokens[k])))
            tokens.extend(seg_tokens[k])
        if not tokens:
            return {}
        lp, fs = log_probs_for(ws, we)
        if lp is None:
            return None
        res = ctc_align(lp, tokens + list(extra_tokens), blank_id)
        if res is None:
            return None
        first, end, best = res
        T = lp.shape[0]
        out = {}
        for k, (a, b) in zip(seg_ids, offsets):
            if a == b:
                continue
            real = [best[t] for t in range(a, b) if seg_tokens[k][t - a] != star_id]
            gap = float(max((first[t + 1] - end[t] for t in range(a, b - 1)), default=0)) * fs
            out[k] = {
                "span": [float(ws + first[a] * fs), float(ws + end[b - 1] * fs)],
                "conf": float(np.mean(np.exp(real))) if real else None,
                "head": float(first[a] * fs),
                "tail": float((T - end[b - 1]) * fs),
                "gap": gap,
                "tok_times": [(float(ws + first[t] * fs), float(ws + end[t] * fs)) for t in range(a, b)],
            }
        return out

    spans = [None] * n
    confs = [None] * n
    tok_times = [None] * n
    gaps = [0.0] * n
    used_wide = [False] * n
    need_wide = [False] * n

    # ---------- 第一輪：逐句，只在 Whisper 時間附近找 ----------
    for k, s in enumerate(segments):
        if seg_tokens[k]:
            ws = max(0.0, s["start"] - pad_sec)
            we = min(duration, s["end"] + pad_sec)
            r = None
            try:
                r = run_align([k], ws, we)
            except Exception as e:
                emit("LOG", f"第 {k + 1} 條第一輪對齊出錯（{e}）")
            if not r or k not in r:
                need_wide[k] = True
            else:
                info = r[k]
                spans[k], confs[k], gaps[k] = info["span"], info["conf"], info["gap"]
                tok_times[k] = info["tok_times"]
                stuck = (info["head"] < edge_sec and ws > 0) or (info["tail"] < edge_sec and we < duration)
                weak = info["conf"] is not None and info["conf"] < low_conf
                need_wide[k] = stuck or weak
        emit("PROGRESS", f"{int(700 * (k + 1) / max(n, 1))} 1000")

    good = [spans[k] is not None and not need_wide[k] for k in range(n)]

    # 找出連續有問題的句子（中間夾著沒有可對齊文字的句子也算在同一段）
    runs, cur = [], []
    for k in range(n):
        if need_wide[k] or (cur and not seg_tokens[k]):
            cur.append(k)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    runs = [[k for k in r if need_wide[k] or seg_tokens[k]] for r in runs]

    n_need = sum(need_wide)
    emit("LOG", f"第一輪：{n - n_need}/{n} 條在 Whisper 時間附近就對好了"
         + (f"，{n_need} 條需要擴大範圍重對" if n_need else ""))

    # ---------- 第二輪：有問題的句子擴大範圍，夾在前後對好的句子之間 ----------
    done_need = 0
    n_stretched = 0
    for run in runs:
        a, b = run[0], run[-1]
        prev_good_end = next((spans[k][1] for k in range(a - 1, -1, -1) if good[k]), 0.0)
        next_good_start = next((spans[k][0] for k in range(b + 1, n) if good[k]), duration)
        lower = max(prev_good_end, segments[a]["start"] - wide_sec, 0.0)
        upper = min(next_good_start, segments[b]["end"] + wide_sec, duration)

        wide = {}
        bound = lower
        i = 0
        while i < len(run):
            j = i + 1
            while j < len(run) and segments[run[j]]["end"] - segments[run[i]]["start"] <= batch_sec:
                j += 1
            batch = run[i:j]
            ws = max(bound, segments[batch[0]]["start"] - wide_sec, 0.0)
            if j < len(run):
                la = segments[run[j]]
                extra = seg_tokens[run[j]][:LOOKAHEAD_TOKENS]
                we = min(upper, max(segments[batch[-1]]["end"] + wide_sec,
                                    min(la["end"], la["start"] + 3.0) + 1.0))
            else:
                extra = []
                we = upper
            r = None
            try:
                r = run_align(batch, ws, we, extra)
            except Exception as e:
                emit("LOG", f"第 {batch[0] + 1}～{batch[-1] + 1} 條第二輪對齊出錯（{e}）")
            if r:
                wide.update(r)
                confident_ends = [v["span"][1] for v in r.values()
                                  if v["conf"] is None or v["conf"] >= low_conf]
                if confident_ends:
                    bound = max(bound, max(confident_ends))
            done_need += sum(1 for k in batch if need_wide[k])
            emit("PROGRESS", f"{700 + int(300 * done_need / max(n_need, 1))} 1000")
            i = j

        for k in run:
            if k not in wide:
                continue
            w = wide[k]
            old_conf = confs[k] if confs[k] is not None else 0.0
            new_conf = w["conf"] if w["conf"] is not None else 0.0
            stretched = w["gap"] > max_inner_gap and w["gap"] > gaps[k] + 0.3
            if spans[k] is None or (new_conf >= old_conf - 0.1 and not stretched):
                spans[k], confs[k] = w["span"], w["conf"]
                tok_times[k] = w["tok_times"]
                used_wide[k] = True
            elif stretched:
                n_stretched += 1

    for k in range(n):
        if seg_tokens[k] and spans[k] is None:
            emit("LOG", f"第 {k + 1} 條對不上，保留原時間：{segments[k]['text']!r}")
    if n_need:
        emit("LOG", f"第二輪：{sum(used_wide)} 條採用擴大範圍的結果"
             + (f"，{n_stretched} 條因為句子被拉長而保留第一輪結果" if n_stretched else ""))

    # 每個詞的時間：詞的第一個字的起點 → 最後一個字的終點；詞裡沒有會發音的字（純標點）→ null
    word_spans = []
    for k in range(n):
        if spans[k] is None or tok_times[k] is None:
            word_spans.append(None)
            continue
        tt = tok_times[k]
        word_spans.append([[tt[a][0], tt[b - 1][1]] if b > a else None
                           for a, b in seg_word_ranges[k]])

    emit("PROGRESS", "1000 1000")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"spans": spans, "confs": confs, "wide": used_wide, "word_spans": word_spans}, f)


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        missing = getattr(e, "name", None) or "transformers"
        emit("ERROR", f"缺少套件 {missing}（請執行 pip install {missing}）")
        sys.exit(2)
    except Exception as e:
        traceback.print_exc()
        sys.stdout.flush()
        emit("ERROR", f"{type(e).__name__}: {e}")
        sys.exit(1)
