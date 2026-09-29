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

用聲音能量微調每個詞的起訖 + 交叉驗證（params.refine.enabled 為真時）：
    CTC 的字通常落在母音附近：詞的起點常晚 0.03～0.1 秒（子音已經開始了）、終點常早 0.05～0.15 秒（尾音還沒結束）。
    對齊完之後，再用聲音能量（每 5 毫秒一格，完全獨立於文字與模型）在每個詞的起訖附近找「靜音 → 有聲」和
    「有聲 → 靜音」的位置，把邊界貼過去。動作很保守：
        ・只有邊界前後找得到明確的靜音／有聲交界才動，而且有幅度上限（起點最多提早 0.15 秒等），
          不會越過前一個詞的終點與下一個詞的起點
        ・連續語音（沒有停頓可貼）、背景聲太大（對比不夠）時維持 CTC 的時間
    同時它就是交叉驗證：每個詞的起點、終點各記一個狀態（result.json 的 checks）——
        ok 兩種算法一致（差距在 agree_sec 內）／moved 依聲音修正／cont 連續語音無法判定／
        noisy 背景聲太大無法判定／silent CTC 說有字的地方卻聽不到聲音（可能是 Whisper 幻覺）
    處理完會在日誌印出一致率、修正量的統計。

用法：
    python align_worker.py --serve <閒置秒數>
        常駐模式（w1_1.py 用這個）：工作一行一個從 stdin 讀進來
        （JSON：{"job": job.json 路徑, "result": result.json 路徑}），開始做印 "START"，做完印 "END <0=成功>"。
        模型載入一次就重複用；每個工作做完先把模型搬到 CPU 釋放顯存，讓 GPU 留給 Whisper／Demucs。
        閒置超過指定秒數自己結束（釋放記憶體）；指定 0 表示做完一個工作就結束。
    python align_worker.py job.json result.json
        單次模式（手動測試用）：做完一個工作就結束。
    job.json    {"audio_npy", "segments": [{"start","end","text","words"(可省略)}, ...],
                 "model_name", "device", "sample_rate",
                 "params": {"pad_sec", "edge_sec", "wide_sec", "batch_sec", "low_conf", "max_inner_gap",
                            "refine": {"enabled", "back_sec", ...}(可省略＝不微調)}}
                 words = 這句拆成詞的文字 list，接起來要等於 text；沒給就當整句是一個詞
    result.json {"spans": [[起點秒, 終點秒] 或 null, ...],
                 "confs": [0～1 或 null, ...], "wide": [這句是否用第二輪結果, ...],
                 "word_spans": [[[詞起點, 詞終點] 或 null（詞裡沒有會發音的字）, ...] 或 null, ...],
                 "checks": [[[起點狀態, 起點修正秒數, 終點狀態, 終點修正秒數] 或 null, ...每個詞] 或 null, ...]}
stdout 每行一則訊息給主程式：
    "LOG <文字>"、"PROGRESS <已完成> <總數>"、"ERROR <文字>"、（常駐模式）"START"、"END <代碼>"
"""

import json
import queue
import sys
import threading
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
# === 用聲音能量微調邊界 + 交叉驗證 ===
# =========================================================
# 為什麼需要：CTC 的字只會在「模型確定聽到那個字」的一兩個 frame 冒出來（很尖的峰），
# 位置通常落在母音附近，而不是子音開始的地方；字唸完之後拖長的尾音也不算在內。
# 所以只靠 CTC，詞的起點常常晚 0.03～0.1 秒、終點常常早 0.05～0.15 秒。
# 聲音能量（有沒有聲音、什麼時候開始有）是完全獨立的另一種量測，不會有這種偏差：
#   ・用它把邊界貼到聲音真的開始／結束的位置（只在邊界前後找得到明確的「靜音→有聲」時才動）
#   ・同時它就是交叉驗證：CTC 跟聲音兩種算法差多少、有沒有一致，都記下來回報
ENV_WIN_SEC = 0.02      # 能量的計算視窗
ENV_HOP_SEC = 0.005     # 每 5 毫秒算一格（wav2vec2 一格 20 毫秒，這裡細很多）
PRE_EMPHASIS = 0.97     # 預強調：壓低低頻（背景音樂的低音、風聲），突顯子音的高頻
MIN_UNIT_SEC = 0.02     # 微調之後一個詞至少要有這麼長，否則放棄微調

REFINE_DEFAULTS = {
    "back_sec": 0.15,          # 起點最多往前（提早）找幾秒
    "fwd_sec": 0.06,           # 起點最多往後（延遲）找幾秒（CTC 起點落在還沒有聲音的地方時）
    "end_fwd_sec": 0.30,       # 終點最多往後（延後）找幾秒，找聲音真正結束的地方
    "end_back_sec": 0.08,      # 終點最多往前（提早）找幾秒（CTC 終點已經落在靜音裡時）
    "min_contrast_db": 15.0,   # 附近「最大聲」跟「背景」至少差幾 dB 才判定得了（背景聲太大就放棄）
    "thr_frac": 0.30,          # 有聲／無聲的門檻：背景到最大聲之間（dB）的 30% 處
    "max_range_db": 30.0,      # 門檻至少要比最大聲低這麼多以內（背景近乎全靜音時，不要把殘留的雜訊當成聲音）
    "dip_sec": 0.03,           # 聲音中間短暫掉下去（例如「っ」前的閉鎖）不超過這麼久，還算同一段聲音
    "agree_sec": 0.04,         # CTC 跟聲音的差距不超過這麼多秒 → 視為「一致」
    "ctx_sec": 1.0,            # 估背景與最大聲時，詞前後各看幾秒
    "quiet_db": 20.0,          # 詞附近整片都比一般說話音量小這麼多 dB → 判定「該處聽不到聲音」
}

STATUS_LABELS = {"ok": "一致", "moved": "依聲音修正", "silent": "該處聽不到聲音",
                 "cont": "連續語音", "noisy": "背景聲太大或沒有停頓"}


def energy_envelope_db(audio, sr: int) -> np.ndarray:
    """每 5 毫秒一格的短時能量（dB，只有相對意義）。分段計算，長影片也不會吃太多記憶體。"""
    win, hop = int(round(ENV_WIN_SEC * sr)), int(round(ENV_HOP_SEC * sr))
    n_frames = (len(audio) - win) // hop + 1 if len(audio) >= win else 0
    env = np.empty(n_frames, dtype=np.float32)
    block = 4000  # 每批 4000 格 = 20 秒
    for f0 in range(0, n_frames, block):
        f1 = min(f0 + block, n_frames)
        s0, s1 = f0 * hop, (f1 - 1) * hop + win
        x = np.asarray(audio[max(s0 - 1, 0):s1], dtype=np.float32)
        y = np.empty(s1 - s0, dtype=np.float32)
        if s0 == 0:
            y[0] = x[0]
            y[1:] = x[1:] - PRE_EMPHASIS * x[:-1]
        else:
            y[:] = x[1:] - PRE_EMPHASIS * x[:-1]
        frames = np.lib.stride_tricks.sliding_window_view(y, win)[::hop]
        env[f0:f1] = 10.0 * np.log10(np.mean(frames * frames, axis=1) + 1e-10)
    return env


def _find_onset(env, i0: int, lo_i: int, fwd_i: int, thr: float, dip_n: int):
    """
    在能量序列 env 上找「聲音開始」的位置。i0 = CTC 給的位置；lo_i = 往前最多找到哪一格；
    fwd_i = 往後最多找到哪一格。回傳 (第一格有聲音的位置, 種類)：
        "edge"   找到明確的「靜音 → 有聲」，回傳的是有聲那一段的第一格
        "cont"   i0 有聲音，但一路往前找到下限都沒有中斷（前面是連續的聲音，可能是上一個詞、背景音）→ 沒有證據
        "silent" i0 沒有聲音，往後找也沒有 → CTC 說有字的地方聽不到聲音
    終點也用同一個函式：把 env 前後反轉再呼叫，「開始」就變成「結束」。
    """
    if env[i0:i0 + 3].max() >= thr:
        first, dips, j = i0, 0, i0 - 1
        while j >= lo_i:
            if env[j] >= thr:
                first, dips = j, 0
            else:
                dips += 1
                if dips >= dip_n:
                    return first, "edge"
            j -= 1
        return (first, "edge") if dips else (first, "cont")  # dips > 0：下限前已經有一小段靜音
    for j in range(i0 + 1, fwd_i + 1):  # CTC 位置還沒有聲音：往後找聲音真正開始的地方
        if env[j] >= thr:
            return j, "edge"
    return i0, "silent"


def _refine_one(env, s: float, e: float, lo_t, hi_t, c: dict, ref_level=None):
    """
    微調一個詞的起訖。lo_t / hi_t = 起點不能早於、終點不能晚於的時間（前一個詞的終點、下一個詞的起點；None = 沒有限制）。
    ref_level = 整份音訊裡一般說話的音量（dB）：這個詞附近整片都比它安靜很多 → 那裡根本沒有人在說話（silent）。
    """
    hop, half = ENV_HOP_SEC, ENV_WIN_SEC / 2
    n = len(env)

    def idx(t):  # 中心點最接近時間 t 的那一格
        return min(max(int(round((t - half) / hop)), 0), n - 1)

    def tm(j):
        return j * hop + half

    bad = (s, e, ["noisy", None, "noisy", None])
    if n == 0 or e <= s:
        return bad
    ctx = env[idx(s - c["ctx_sec"]): idx(e + c["ctx_sec"]) + 1]
    level, floor = float(np.percentile(ctx, 95)), float(np.percentile(ctx, 10))
    contrast = level - floor
    if contrast < c["min_contrast_db"]:
        if ref_level is not None and level < ref_level - c["quiet_db"]:  # 附近整片安靜，CTC 卻說這裡有字
            return s, e, ["silent", None, "silent", None]
        return bad
    thr = max(floor + c["thr_frac"] * contrast, level - c["max_range_db"], floor + 6.0)
    thr = min(thr, level - 3.0)
    dip_n = max(1, int(round(c["dip_sec"] / hop)))

    # ---- 起點 ----
    lo_eff = s - c["back_sec"] if lo_t is None else min(max(lo_t, s - c["back_sec"]), s)
    i_s = idx(s)
    j, kind = _find_onset(env, i_s, min(idx(lo_eff), i_s), max(idx(min(s + c["fwd_sec"], e - MIN_UNIT_SEC)), i_s),
                          thr, dip_n)
    s2, s_status, s_delta = s, kind, None
    if kind == "edge":
        s2 = s if j == i_s else max(tm(j), lo_eff)
        s_delta = s2 - s
        s_status = "ok" if abs(s_delta) <= c["agree_sec"] else "moved"

    # ---- 終點：把 env 前後反轉，用同一個函式找「聲音結束」----
    hi_eff = e + c["end_fwd_sec"] if hi_t is None else max(min(hi_t, e + c["end_fwd_sec"]), e)
    i_e = idx(e)
    top = max(idx(hi_eff), i_e)
    bottom = min(idx(max(e - c["end_back_sec"], s2 + MIN_UNIT_SEC)), i_e)
    rev = env[bottom:top + 1][::-1]
    j, kind = _find_onset(rev, top - i_e, 0, top - bottom, thr, dip_n)
    e2, e_status, e_delta = e, kind, None
    if kind == "edge":
        e2 = e if top - j == i_e else min(tm(top - j), hi_eff)
        e_delta = e2 - e
        e_status = "ok" if abs(e_delta) <= c["agree_sec"] else "moved"

    if e2 < s2 + MIN_UNIT_SEC:  # 兩邊各自合理、合起來卻不成立（極端情況）：整個放棄，維持 CTC 的時間
        return bad
    return s2, e2, [s_status, s_delta, e_status, e_delta]


def refine_boundaries(env, times: list, cfg=None):
    """
    times = [(起點秒, 終點秒), ...]：照時間順序排好的詞。回傳 (微調後的 times, checks)，
    checks[i] = [起點狀態, 起點修正秒數, 終點狀態, 終點修正秒數]，狀態見 STATUS_LABELS。
    一個詞的邊界不會越過前一個詞（已微調過）的終點、以及下一個詞（CTC）的起點。
    """
    c = {**REFINE_DEFAULTS, **(cfg or {})}
    ref_level = None
    if len(env):
        last = len(env) - 1
        levels = [np.percentile(env[min(max(int(round((s - ENV_WIN_SEC / 2) / ENV_HOP_SEC)), 0), last):
                                    min(max(int(round((e - ENV_WIN_SEC / 2) / ENV_HOP_SEC)), 0), last) + 1], 90)
                  for s, e in times if e > s]
        ref_level = float(np.median(levels)) if levels else None
    out, checks = [], []
    prev_end = None
    for i, (s, e) in enumerate(times):
        next_start = times[i + 1][0] if i + 1 < len(times) else None
        s2, e2, check = _refine_one(env, s, e, prev_end, next_start, c, ref_level)
        out.append((s2, e2))
        checks.append(check)
        prev_end = e2
    return out, checks


def summarize_checks(flat_checks: list) -> list[str]:
    """把 checks 統計成幾行文字（給日誌）：一致率、修正量、無法判定的原因。"""
    lines = []
    for label, si in (("起點", 0), ("終點", 2)):
        statuses = [c[si] for c in flat_checks]
        count = {k: statuses.count(k) for k in STATUS_LABELS}
        judged = count["ok"] + count["moved"] + count["silent"]
        if not statuses:
            continue
        parts = []
        if judged:
            parts.append(f"一致 {count['ok']} 個（{100 * count['ok'] / judged:.0f}%）")
            if count["moved"]:
                deltas = np.array([c[si + 1] for c in flat_checks if c[si] == "moved"])
                parts.append(f"依聲音修正 {count['moved']} 個（{100 * count['moved'] / judged:.0f}%，"
                             f"中位數 {np.median(deltas):+.2f} 秒，最大 {deltas[np.abs(deltas).argmax()]:+.2f} 秒）")
            if count["silent"]:
                parts.append(f"該處聽不到聲音 {count['silent']} 個")
        skipped = [f"{STATUS_LABELS[k]} {count[k]}" for k in ("cont", "noisy") if count[k]]
        if skipped:
            parts.append("無法判定：" + "、".join(skipped))
        lines.append(f"{label}：" + "；".join(parts))
    return lines


# =========================================================
# === 主流程 ===
# =========================================================
class Aligner:
    """wav2vec2 模型。載入一次可以重複對很多個工作用（常駐模式）。"""

    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.torch = torch
        self.requested = (model_name, device)  # 之後的工作要求不同的模型或裝置時才需要重載
        self.jobs_done = 0
        if str(device) == "auto":
            device = "cuda"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            emit("LOG", "偵測不到可用的 CUDA GPU（或 PyTorch 裝成 CPU 版），改用 CPU 對齊，速度會慢很多")
            device = "cpu"
        self.device = str(device)

        emit("LOG", f"載入對齊模型：{model_name} …（第一次會下載，約 1.2 GB）")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()
        tokenizer = self.processor.tokenizer
        self.vocab = tokenizer.get_vocab()
        self.blank_id = tokenizer.pad_token_id  # wav2vec2 CTC 的 blank 就是 <pad>
        self.delim_id = self.vocab.get("|")
        self.star_id = max(self.vocab.values()) + 1  # 萬用字元：接在字表最後面的額外一欄
        # 萬用字元「任何發音都可以」是從這些欄裡取最高機率。只算真的會發音的字：
        # 不含 blank、詞分隔符 |、以及 <s> </s> <unk> 這些特殊符號（它們不是發音，取到了會讓萬用字元亂貼）
        excluded = {self.blank_id, self.delim_id, *getattr(tokenizer, "all_special_ids", [])}
        candidates = [v for v in sorted(set(self.vocab.values())) if v not in excluded]
        if not candidates:  # 字表很怪、全被排除時退回只排除 blank
            candidates = [v for v in sorted(set(self.vocab.values())) if v != self.blank_id]
        self.star_candidates = np.array(candidates)
        # 每個 frame 對應幾個取樣點（wav2vec2 是 320 = 0.02 秒）。取不到時 log_probs 會退回用視窗長度估
        stride = getattr(getattr(self.model, "config", None), "conv_stride", None)
        try:
            self.frame_stride = int(np.prod(stride)) if stride else None
        except (TypeError, ValueError):
            self.frame_stride = None

    def park(self) -> None:
        """工作做完：把模型搬到 CPU、釋放顯存。常駐等下一個檔案的期間，GPU 留給 Whisper／Demucs。"""
        if self.device.startswith("cuda"):
            self.model.to("cpu")
            self.torch.cuda.empty_cache()

    def unpark(self) -> None:
        """下一個工作開始：把模型搬回 GPU。"""
        if self.device.startswith("cuda"):
            self.model.to(self.device)

    def _posteriors(self, chunk: np.ndarray, sr: int) -> np.ndarray:
        """跑模型，回傳每個 frame 對每個字的 log 機率 (T, 字表大小)。"""
        torch = self.torch
        inputs = self.processor(chunk, sampling_rate=sr, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(inputs.input_values.to(self.device)).logits[0]
        return torch.log_softmax(logits.float(), dim=-1).cpu().numpy()

    def log_probs(self, chunk: np.ndarray, sr: int):
        """
        對一段音訊跑 wav2vec2，回傳 (log 機率含萬用字元欄, 每 frame 秒數)。
        第 t 個 frame 一定是從 t * 步幅 個取樣點開始（wav2vec2 是 320 個 = 0.02 秒）。
        以前用「視窗秒數 / frame 數」估每格多久：因為最後一個不足一格的尾巴（加上 400 取樣的感受野）
        會被平均分給每一格，每格多估了一點點，越靠近視窗尾端越晚，最多晚一整格（20 毫秒）。
        """
        lp = self._posteriors(chunk, sr)
        if lp.shape[1] <= self.star_id:  # 模型輸出欄數跟字表不一致時補齊
            lp = np.pad(lp, ((0, 0), (0, self.star_id - lp.shape[1])), constant_values=-np.inf)
        star = lp[:, self.star_candidates[self.star_candidates < lp.shape[1]]].max(axis=1, keepdims=True) - STAR_PENALTY
        lp = np.concatenate([lp[:, :self.star_id], star], axis=1)
        frame_sec = (len(chunk) / sr) / lp.shape[0]  # 估的：模型步幅不明、或跟實際 frame 數對不上時才用
        if self.frame_stride and abs(len(chunk) / self.frame_stride - lp.shape[0]) <= 3:
            frame_sec = self.frame_stride / sr
        return lp, frame_sec


def align_job(aligner: Aligner, job: dict) -> dict:
    """用載入好的模型做一個對齊工作（兩輪對齊，說明見檔案開頭），回傳要寫進 result.json 的內容。"""
    sr = int(job["sample_rate"])
    segments = job["segments"]
    p = job["params"]
    pad_sec = float(p["pad_sec"])        # 第一輪：Whisper 時間前後各多取幾秒
    edge_sec = float(p["edge_sec"])      # 對齊結果離搜尋範圍邊緣小於這個值 → 視為被範圍卡住
    wide_sec = float(p["wide_sec"])      # 第二輪：往前後各多找幾秒
    batch_sec = float(p["batch_sec"])    # 第二輪：連續有問題的句子每批最長幾秒
    low_conf = float(p["low_conf"])
    max_inner_gap = float(p["max_inner_gap"])  # 同一句相鄰兩字最多空幾秒，超過視為被拉長
    vocab, blank_id, delim_id, star_id = aligner.vocab, aligner.blank_id, aligner.delim_id, aligner.star_id

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
    if aligner.jobs_done == 0:
        emit("LOG", f"對齊模型載入完成（字表 {len(vocab)} 個字元），開始對齊 {n} 條字幕")
    else:
        emit("LOG", f"沿用已載入的對齊模型，開始對齊 {n} 條字幕")
    if unknown_chars:
        shown = "".join(sorted(unknown_chars)[:40])
        more = f" 等 {len(unknown_chars)} 個" if len(unknown_chars) > 40 else ""
        emit("LOG", f"字表裡沒有、改用萬用字元對齊的字：{shown}{more}")

    def log_probs_for(win_start: float, win_end: float):
        """取一段音訊跑 wav2vec2，回傳 (log 機率含萬用字元欄, 每 frame 秒數)。"""
        chunk = np.asarray(audio[int(win_start * sr): int(win_end * sr)], dtype=np.float32)
        if len(chunk) < int(0.1 * sr):
            return None, 0.0
        return aligner.log_probs(chunk, sr)

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

    # ---------- 用聲音能量微調每個詞的起訖，並交叉驗證（說明見上面「用聲音能量微調邊界」）----------
    checks = [None] * n
    refine_cfg = p.get("refine")
    if refine_cfg and refine_cfg.get("enabled"):
        units = []  # (句子編號, 第幾個詞, 詞的第一個 token, 詞的最後一個 token + 1)，照時間順序
        for k in range(n):
            if spans[k] is None or tok_times[k] is None:
                continue
            tok_times[k] = [list(t) for t in tok_times[k]]
            checks[k] = [None] * len(seg_word_ranges[k])
            units.extend((k, wi, a, b) for wi, (a, b) in enumerate(seg_word_ranges[k]) if b > a)
        if units:
            env = energy_envelope_db(audio, sr)
            new_times, unit_checks = refine_boundaries(
                env, [(tok_times[k][a][0], tok_times[k][b - 1][1]) for k, _, a, b in units], refine_cfg)
            for (k, wi, a, b), (s, e), check in zip(units, new_times, unit_checks):
                tok_times[k][a][0], tok_times[k][b - 1][1] = s, e
                checks[k][wi] = check
            for k in {u[0] for u in units}:
                spans[k] = [tok_times[k][0][0], tok_times[k][-1][1]]
            emit("LOG", f"[對齊驗證] 用聲音能量檢查 {len(units)} 個詞的起訖（CTC 對齊 vs 聲音開始／結束的位置）：")
            for line in summarize_checks(unit_checks):
                emit("LOG", "    " + line)

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
    aligner.jobs_done += 1
    return {"spans": spans, "confs": confs, "wide": used_wide, "word_spans": word_spans, "checks": checks}


def run_once(job_path: str, result_path: str) -> None:
    """單次模式：讀一個工作、做完、寫結果。"""
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    result = align_job(Aligner(job["model_name"], job["device"]), job)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


def serve(idle_sec: float) -> None:
    """常駐模式：從 stdin 一行一個工作，模型只載入一次；idle_sec <= 0 表示做完一個就結束。"""
    lines: queue.Queue = queue.Queue()

    def read_stdin() -> None:
        for line in sys.stdin:
            lines.put(line)
        lines.put(None)  # stdin 被關掉了：主程式不需要我了

    threading.Thread(target=read_stdin, daemon=True).start()
    aligner = None
    while True:
        try:
            line = lines.get(timeout=idle_sec) if idle_sec > 0 else lines.get()
        except queue.Empty:
            emit("LOG", f"閒置超過 {idle_sec:g} 秒，對齊程序結束（下次要對齊會重新載入模型）")
            return
        if line is None:
            return
        if not line.strip():
            continue
        request = json.loads(line)
        emit("START", "")
        try:
            with open(request["job"], encoding="utf-8") as f:
                job = json.load(f)
            key = (job["model_name"], job["device"])
            if aligner is not None and aligner.requested == key:
                aligner.unpark()
            else:
                aligner = None  # 先放掉舊的再載入新的
                aligner = Aligner(*key)
            result = align_job(aligner, job)
            with open(request["result"], "w", encoding="utf-8") as f:
                json.dump(result, f)
            aligner.park()
        except ImportError as e:
            missing = getattr(e, "name", None) or "transformers"
            emit("ERROR", f"缺少套件 {missing}（請執行 pip install {missing}）")
            emit("END", 2)
            sys.exit(2)
        except Exception as e:
            traceback.print_exc()
            sys.stdout.flush()
            emit("ERROR", f"{type(e).__name__}: {e}")
            emit("END", 1)
            sys.exit(1)  # 出錯之後模型狀態不可信（例如顯存不足），結束讓主程式下次重開
        emit("END", 0)
        if idle_sec <= 0:
            return


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--serve":
        serve(float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
    else:
        run_once(sys.argv[1], sys.argv[2])


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
