"""聲音能量微調邊界 + 交叉驗證（align_worker.py）。
用 tests/synth.py 合成的音訊：詞的真正起訖是已知的，可以驗證微調把 CTC 偏掉的邊界貼回真正的位置，
背景太吵或連續語音時不亂動，而且絕不越過鄰居。"""
import json

import numpy as np
import pytest

from helpers import make_align_job
from synth import SR, synth_speech

HOP = 0.005


def ctc_like(words, lag=0.02, early=0.08):
    """模擬 CTC 的偏差：起點落在母音開始之後（晚 lag），終點在聲音真正結束之前（早 early）。"""
    return [(on + cons + lag, max(off - early, on + cons + lag + 0.05)) for on, off, cons in words]


# ---------------- 能量曲線 ----------------
def test_envelope_follows_amplitude_in_db(worker_module):
    x = np.zeros(SR * 2, dtype=np.float32)
    x[SR // 2:SR] = 0.1 * np.random.default_rng(0).normal(size=SR // 2)      # 0.5～1.0 秒
    x[SR:SR + SR // 2] = 1.0 * np.random.default_rng(1).normal(size=SR // 2)  # 1.0～1.5 秒，大 10 倍 = +20 dB
    env = worker_module.energy_envelope_db(x, SR)
    quiet, loud, silent = env[int(0.7 / HOP)], env[int(1.2 / HOP)], env[int(0.2 / HOP)]
    assert loud - quiet == pytest.approx(20.0, abs=1.5)
    assert silent < quiet - 60                      # 完全靜音 = 極低（下限 -100 dB）


def test_envelope_length_and_block_boundaries(worker_module):
    """分段計算（每批 20 秒）的結果要跟一次算完的一樣，批與批交界處也不能有斷層。"""
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.05, SR * 47).astype(np.float32)
    env = worker_module.energy_envelope_db(x, SR)
    win, hop = int(0.02 * SR), int(HOP * SR)
    assert len(env) == (len(x) - win) // hop + 1
    y = np.concatenate(([x[0]], x[1:] - worker_module.PRE_EMPHASIS * x[:-1])).astype(np.float32)
    for f in (0, 1, 3999, 4000, 4001, 7999, 8000, len(env) - 1):   # 批的交界前後
        expected = 10 * np.log10(np.mean(y[f * hop:f * hop + win] ** 2) + 1e-10)
        assert env[f] == pytest.approx(expected, abs=1e-3)


def test_envelope_of_very_short_audio_is_empty(worker_module):
    assert len(worker_module.energy_envelope_db(np.zeros(100, dtype=np.float32), SR)) == 0


def test_envelope_reads_memory_mapped_audio(worker_module, tmp_path):
    x = synth_speech(3, [(1.0, 1.5, 0.0)])
    np.save(tmp_path / "a.npy", x)
    mm = np.load(tmp_path / "a.npy", mmap_mode="r")
    np.testing.assert_allclose(worker_module.energy_envelope_db(mm, SR), worker_module.energy_envelope_db(x, SR))


# ---------------- 找聲音開始的位置 ----------------
def onset(worker_module, env, i0, lo=0, fwd=None, thr=-40.0, dip=6):
    return worker_module._find_onset(np.array(env, dtype=float), i0, lo, len(env) - 1 if fwd is None else fwd, thr, dip)


QUIET, LOUD = -80.0, -20.0


def test_onset_edge_walks_back_to_the_start_of_the_sound(worker_module):
    env = [QUIET] * 20 + [LOUD] * 30 + [QUIET] * 20        # 聲音在 20～49
    assert onset(worker_module, env, 35) == (20, "edge")   # CTC 落在聲音中間 → 貼到聲音開頭
    assert onset(worker_module, env, 20) == (20, "edge")   # 已經在開頭 → 不動


def test_onset_continuous_sound_is_not_judged(worker_module):
    env = [LOUD] * 60
    assert onset(worker_module, env, 40, lo=25) == (25, "cont")   # 一路走到下限都沒有中斷


def test_onset_tolerates_a_short_dip_but_not_a_long_one(worker_module):
    short = [QUIET] * 10 + [LOUD] * 10 + [QUIET] * 4 + [LOUD] * 10 + [QUIET] * 5     # 中間掉 4 格 = 20 毫秒
    long = [QUIET] * 10 + [LOUD] * 10 + [QUIET] * 8 + [LOUD] * 10 + [QUIET] * 5      # 掉 8 格 = 40 毫秒
    assert onset(worker_module, short, 28) == (10, "edge")   # 還算同一段聲音，回到最前面
    assert onset(worker_module, long, 28) == (28, "edge")    # 靜音夠長，就是兩段聲音：只貼到後一段的開頭


def test_onset_stops_at_the_lower_limit_with_a_gap_before_it(worker_module):
    env = [LOUD] * 10 + [QUIET] * 3 + [LOUD] * 20     # 下限之前先出現 3 格靜音（不到 dip 門檻）
    assert onset(worker_module, env, 25, lo=10) == (13, "edge")


def test_onset_moves_forward_when_ctc_is_too_early(worker_module):
    env = [QUIET] * 30 + [LOUD] * 30
    assert onset(worker_module, env, 24, fwd=40) == (30, "edge")
    assert onset(worker_module, env, 24, fwd=27) == (24, "silent")   # 往後找的範圍不夠 → 聽不到


def test_onset_silent_when_nothing_is_there(worker_module):
    assert onset(worker_module, [QUIET] * 50, 25, fwd=35) == (25, "silent")


# ---------------- 微調整串詞 ----------------
WORDS = [(1.0, 1.5, 0.06), (1.62, 2.0, 0.0), (3.0, 3.4, 0.08), (3.45, 3.9, 0.05), (5.0, 5.2, 0.04)]


@pytest.fixture
def clean_env(worker_module):
    return worker_module.energy_envelope_db(synth_speech(7, WORDS), SR)


def test_refine_snaps_late_starts_and_early_ends_back_to_the_real_edges(worker_module, clean_env):
    ctc = ctc_like(WORDS)
    new, checks = worker_module.refine_boundaries(clean_env, ctc)
    for (on, off, _), (s, e) in zip(WORDS, new):
        assert abs(s - on) <= 0.015 and abs(e - off) <= 0.015      # 貼回真正的起訖
    before = np.mean([abs(a - on) + abs(b - off) for (on, off, _), (a, b) in zip(WORDS, ctc)])
    after = np.mean([abs(s - on) + abs(e - off) for (on, off, _), (s, e) in zip(WORDS, new)])
    assert before > 0.1 and after < 0.02                            # 平均誤差從 0.1 秒以上降到 0.02 秒以內
    # 第二個詞沒有子音，CTC 只晚了 0.02 秒（在「一致」的 0.04 以內）；其他詞晚了 0.06～0.1 秒
    assert [c[0] for c in checks] == ["moved", "ok", "moved", "moved", "moved"] and all(c[2] == "moved" for c in checks)
    assert all(c[1] < 0 for c in checks) and all(c[3] > 0 for c in checks)   # 起點都提早、終點都延後


def test_refine_reports_the_amount_moved(worker_module, clean_env):
    ctc = ctc_like(WORDS)
    new, checks = worker_module.refine_boundaries(clean_env, ctc)
    for (s0, e0), (s1, e1), (_, ds, _, de) in zip(ctc, new, checks):
        assert ds == pytest.approx(s1 - s0) and de == pytest.approx(e1 - e0)


def test_refine_marks_agreement_when_ctc_is_already_right(worker_module):
    words = [(1.0, 1.5, 0.0), (3.0, 3.4, 0.0)]
    env = worker_module.energy_envelope_db(synth_speech(5, words), SR)
    new, checks = worker_module.refine_boundaries(env, [(1.005, 1.495), (3.005, 3.395)])
    assert [c[0] for c in checks] == ["ok", "ok"] and [c[2] for c in checks] == ["ok", "ok"]
    assert all(abs(c[1]) <= 0.04 and abs(c[3]) <= 0.04 for c in checks)


def test_refine_never_moves_beyond_the_configured_limits(worker_module):
    words = [(2.0, 3.0, 0.0)]
    env = worker_module.energy_envelope_db(synth_speech(5, words), SR)
    # CTC 起點在聲音裡面 0.4 秒處：真正的開頭離它超過 back_sec（0.15）→ 沒有證據，維持原樣
    new, checks = worker_module.refine_boundaries(env, [(2.4, 2.9)])
    assert new[0][0] == 2.4 and checks[0][0] == "cont"
    # 放寬到 0.5 就找得到
    new, checks = worker_module.refine_boundaries(env, [(2.4, 2.9)], {"back_sec": 0.5})
    assert abs(new[0][0] - 2.0) <= 0.015 and checks[0][0] == "moved"


def test_refine_moves_a_too_early_start_later_but_only_slightly(worker_module):
    words = [(2.0, 3.0, 0.0)]
    env = worker_module.energy_envelope_db(synth_speech(5, words), SR)
    new, checks = worker_module.refine_boundaries(env, [(1.96, 2.9)])    # CTC 起點在聲音開始前 0.04 秒
    assert abs(new[0][0] - 2.0) <= 0.015 and checks[0][0] in ("ok", "moved")
    new, checks = worker_module.refine_boundaries(env, [(1.8, 2.9)])     # 早了 0.2 秒：超過 fwd_sec → 不敢動，也不算「聽不到聲音」
    assert new[0][0] == 1.8 and checks[0][0] == "unclear"              # （詞的範圍內明明有聲音，只是起點附近沒有）


def test_refine_flags_words_placed_where_there_is_no_sound(worker_module):
    """CTC 說這裡有字，能量卻是一片靜音（例如 Whisper 幻覺、對到錯的地方）→ silent，時間不動。"""
    words = [(1.0, 1.5, 0.0), (4.0, 4.5, 0.0)]
    env = worker_module.energy_envelope_db(synth_speech(6, words), SR)
    new, checks = worker_module.refine_boundaries(env, [(1.05, 1.4), (2.5, 2.9), (4.05, 4.4)])
    assert new[1] == (2.5, 2.9) and checks[1] == ["silent", None, "silent", None]
    assert checks[0][0] == "moved" and checks[2][0] == "moved"


def test_refine_gives_up_when_the_background_is_too_loud(worker_module):
    words = [(1.0, 1.5, 0.0), (3.0, 3.4, 0.0)]
    for kwargs in ({"floor_db": -32.0}, {"bgm_db": -18.0}):
        env = worker_module.energy_envelope_db(synth_speech(5, words, **kwargs), SR)
        ctc = ctc_like(words)
        new, checks = worker_module.refine_boundaries(env, ctc)
        assert new == ctc and all(c == ["noisy", None, "noisy", None] for c in checks)


def test_refine_copes_with_steady_background_music_below_the_voice(worker_module):
    words = [(1.0, 1.5, 0.06), (3.0, 3.4, 0.04)]
    env = worker_module.energy_envelope_db(synth_speech(5, words, bgm_db=-40.0), SR)
    new, _ = worker_module.refine_boundaries(env, ctc_like(words))
    for (on, off, _), (s, e) in zip(words, new):
        assert abs(s - on) <= 0.02 and abs(e - off) <= 0.02


def test_refine_threshold_setting_trades_sensitivity_for_caution(worker_module):
    """聲音是慢慢變大的（每格 +1 dB）：門檻設得越高，認定「聲音開始」的位置越晚。"""
    env = np.full(400, -80.0)
    env[100:160] = np.linspace(-80.0, -21.0, 60)     # 漸強
    env[160:260] = -20.0
    starts = {}
    for frac in (0.3, 0.6, 0.9):
        cfg = {**worker_module.REFINE_DEFAULTS, "thr_frac": frac, "back_sec": 1.0}
        s, _, check = worker_module._refine_one(env, 1.0, 1.25, None, None, cfg)
        assert check[0] == "moved"
        starts[frac] = s
    assert starts[0.3] < starts[0.6] < starts[0.9]
    assert starts[0.6] - starts[0.3] == pytest.approx(0.03, abs=0.011)          # 門檻高 6 dB ≈ 晚 6 格 = 30 毫秒


def test_refine_ignores_faint_residue_when_the_background_is_digital_silence(worker_module):
    """人聲分離後背景常常近乎全靜音（-100 dB），殘留的一點雜訊（-70 dB 上下）不能被當成「聲音」把起點拉到雜訊裡去。"""
    x = synth_speech(4, [(1.0, 1.5, 0.06)], floor_db=-100.0)
    x[int(0.85 * SR):int(1.0 * SR)] += np.random.default_rng(9).normal(0, 10 ** (-72 / 20), int(0.15 * SR)).astype(np.float32)
    env = worker_module.energy_envelope_db(x, SR)
    new, checks = worker_module.refine_boundaries(env, [(1.08, 1.42)])
    assert abs(new[0][0] - 1.0) <= 0.015 and checks[0][0] == "moved"


def test_refine_leaves_continuous_speech_alone(worker_module):
    """兩個詞之間沒有停頓：沒有「靜音 → 有聲」可以貼，維持 CTC 的時間，不亂猜。"""
    words = [(1.0, 1.4, 0.0), (1.4, 1.8, 0.0), (1.8, 2.2, 0.0)]
    env = worker_module.energy_envelope_db(synth_speech(4, words), SR)
    ctc = [(1.03, 1.35), (1.45, 1.75), (1.85, 2.1)]
    new, checks = worker_module.refine_boundaries(env, ctc)
    assert new[1] == ctc[1]                                   # 中間的詞前後都是連續的聲音
    assert checks[1][0] == "cont" and checks[1][2] == "cont"


def test_refine_start_does_not_reach_back_into_the_previous_word(worker_module):
    """前一個詞的聲音中間有個短停頓（1.44～1.48），後面一路連到下一個詞：下一個詞的起點不能往回貼到那個停頓後面、
    蓋到前一個詞的範圍裡（前一個詞的 CTC 終點是 1.5）。"""
    x = synth_speech(4, [(1.0, 1.44, 0.0), (1.48, 2.1, 0.0)])
    env = worker_module.energy_envelope_db(x, SR)
    new, checks = worker_module.refine_boundaries(env, [(1.0, 1.5), (1.58, 2.0)])
    assert new[1][0] >= new[0][1] and new[1][0] == 1.58 and checks[1][0] == "cont"


def test_refine_end_does_not_run_into_the_next_word(worker_module):
    """下一個詞的 CTC 起點（1.4）其實落在前一個詞聲音的中間：前一個詞的終點不能越過它，不然兩個詞會重疊。"""
    x = synth_speech(4, [(1.0, 1.5, 0.0), (1.7, 2.2, 0.0)])
    env = worker_module.energy_envelope_db(x, SR)
    new, checks = worker_module.refine_boundaries(env, [(1.05, 1.3), (1.4, 1.95)])
    assert new[0][1] <= new[1][0] and new[0][1] == pytest.approx(1.3) and checks[0][2] == "cont"


def test_refine_never_overlaps_neighbours_or_breaks_order(worker_module):
    rng = np.random.default_rng(4)
    words, t = [], 1.0
    for _ in range(60):
        dur = rng.uniform(0.1, 0.5)
        words.append((t, t + dur, float(rng.choice([0.0, 0.04, 0.08]))))
        t += dur + rng.uniform(0.0, 0.3)
    env = worker_module.energy_envelope_db(synth_speech(t + 1, words, seed=3), SR)
    ctc = []
    for on, off, cons in words:                       # 隨機的 CTC 偏差；保證 CTC 自己不重疊、每個詞至少 30 毫秒
        s = on + cons + rng.uniform(-0.03, 0.08)
        e = max(off - rng.uniform(-0.03, 0.12), s + 0.03)
        ctc.append((s, e))
    ctc = [(s, min(e, ctc[i + 1][0] - 0.001) if i + 1 < len(ctc) else e) for i, (s, e) in enumerate(ctc)]
    ctc = [(s, e) for s, e in ctc if e > s + 0.02]
    new, checks = worker_module.refine_boundaries(env, ctc)
    cfg = worker_module.REFINE_DEFAULTS
    assert len(new) == len(ctc) == len(checks)
    for i, ((s0, e0), (s1, e1)) in enumerate(zip(ctc, new)):
        assert e1 > s1                                                       # 沒有反過來
        assert s0 - cfg["back_sec"] - 1e-9 <= s1 <= s0 + cfg["fwd_sec"] + 0.003     # 起點在設定的範圍內
        assert e0 - cfg["end_back_sec"] - 0.003 <= e1 <= e0 + cfg["end_fwd_sec"] + 1e-9   # 終點也是
        if i + 1 < len(new):
            assert e1 <= new[i + 1][0] + 1e-9                                # 不蓋到下一個詞
            assert e1 <= ctc[i + 1][0] + 1e-9                                # 終點也不越過下一個詞原本的起點
    assert sum(c[0] == "moved" for c in checks) > 10                         # 而且不是因為什麼都沒動才成立


def test_refine_falls_back_when_the_result_would_be_shorter_than_the_minimum(worker_module):
    """安全網：兩邊各自找得到、合起來卻讓詞縮成不到 20 毫秒（網格取整數的邊緣情況）→ 整個放棄，維持 CTC 的時間。"""
    env = np.full(400, -80.0)
    env[100:141] = -20.0        # 別處有正常的聲音，才判定得了對比
    env[199:203] = -20.0        # 這個詞附近只有 4 格 = 20 毫秒的聲音（199～202 格）
    s, e, check = worker_module._refine_one(env, 1.0025, 1.035, None, None, {**worker_module.REFINE_DEFAULTS})
    assert (s, e) == (1.0025, 1.035) and check == ["noisy", None, "noisy", None]


def test_refine_handles_degenerate_input(worker_module):
    assert worker_module.refine_boundaries(np.zeros(0), [(1.0, 2.0)])[1] == [["noisy", None, "noisy", None]]
    assert worker_module.refine_boundaries(np.zeros(500), [(1.0, 1.0)])[1] == [["noisy", None, "noisy", None]]
    assert worker_module.refine_boundaries(np.zeros(500), []) == ([], [])
    # 詞在音訊之外（時間超過音訊長度）不能出錯
    env = worker_module.energy_envelope_db(synth_speech(3, [(1.0, 1.5, 0.0)]), SR)
    new, checks = worker_module.refine_boundaries(env, [(10.0, 10.5)])
    assert len(new) == 1


# ---------------- 統計 ----------------
def test_summary_reports_agreement_corrections_and_what_could_not_be_judged(worker_module):
    checks = ([["ok", 0.01, "ok", -0.02]] * 6 + [["moved", -0.06, "moved", 0.10], ["moved", -0.10, "moved", 0.08]]
              + [["silent", None, "silent", None], ["cont", None, "cont", None], ["noisy", None, "noisy", None]])
    lines = worker_module.summarize_checks(checks)
    assert len(lines) == 2 and lines[0].startswith("起點：") and lines[1].startswith("終點：")
    start = lines[0]
    assert "一致 6 個（67%）" in start and "依聲音修正 2 個（22%" in start
    assert "中位數 -0.08 秒" in start and "最大 -0.10 秒" in start
    assert "該處聽不到聲音 1 個" in start and "連續語音 1" in start and "背景聲太大或沒有停頓 1" in start
    assert "+0.10 秒" in lines[1]                                                    # 終點是延後修正


def test_summary_with_nothing_judgeable(worker_module):
    lines = worker_module.summarize_checks([["noisy", None, "noisy", None]] * 3)
    assert lines == ["起點：無法判定：背景聲太大或沒有停頓 3", "終點：無法判定：背景聲太大或沒有停頓 3"]
    assert worker_module.summarize_checks([]) == []


# ---------------- 串起來：align_job（CTC 峰的位置是我們指定的）----------------
def locate(audio, chunk):
    """chunk 是 audio 的哪一段開始（取樣點）。合成音訊有背景雜訊，開頭幾個樣本就足以唯一定位。"""
    for i in np.flatnonzero(audio == chunk[0]):
        if np.array_equal(audio[i:i + 64], chunk[:64]):
            return int(i)
    raise AssertionError("找不到這一段音訊")


class ScriptedCtc:
    """
    假的 wav2vec2：不看音訊內容，直接照劇本在指定的時間放 CTC 的「尖峰」（像真的 CTC 那樣，
    字只出現在一兩個 frame）。劇本 = [(絕對時間秒, token 編號), ...]。
    """

    def __init__(self, worker_module, audio, script):
        self.wm, self.audio, self.script = worker_module, audio, script

    def install(self, aligner):
        def posteriors(chunk, sr):
            offset = locate(self.audio, chunk) / sr
            frames = (len(chunk) - 400) // 320 + 1
            lp = np.full((frames, max(aligner.vocab.values()) + 1), np.log(1e-4))
            lp[:, aligner.blank_id] = np.log(0.97)
            for t, token in self.script:
                f = int(round((t - offset) / 0.02))
                if 0 <= f < frames:
                    lp[f, :] = np.log(1e-4)
                    lp[f, token] = np.log(0.95)
            return lp
        aligner._posteriors = posteriors
        return aligner


def make_scripted_job(worker_module, tmp_path, words_spec, refine, silent_spec=()):
    """
    words_spec = [(文字, [詞的切法], 真正起點, 真正終點, 子音秒數)]。每個字幕一個句子；CTC 的峰依模型的常見偏差
    排在「母音開始之後 0.02 秒」到「真正結束前 0.08 秒」之間。回傳 (job, aligner, 真正的起訖)。
    silent_spec = [(文字, 起點, 終點)]：音訊裡沒有聲音、但 Whisper 說有、CTC 也硬對上的字幕（幻覺）。
    """
    aligner = worker_module.Aligner("x", "cpu")
    speech = [(on, off, cons) for _, _, on, off, cons in words_spec]
    audio = synth_speech(max([off for _, off, _ in speech] + [off for _, _, off in silent_spec]) + 1.5, speech)
    np.save(tmp_path / "audio.npy", audio)
    script, segments = [], []
    for text, words, on, off, cons in words_spec:
        tokens = worker_module.text_to_tokens(text, aligner.vocab, aligner.delim_id, aligner.star_id, set())
        start, end = on + cons + 0.02, off - 0.08
        script += list(zip(np.linspace(start, end - 0.02, len(tokens)), tokens))
        segments.append({"start": round(on + 0.05, 2), "end": round(off - 0.05, 2), "text": text, "words": words})
    for text, on, off in silent_spec:
        tokens = worker_module.text_to_tokens(text, aligner.vocab, aligner.delim_id, aligner.star_id, set())
        script += list(zip(np.linspace(on + 0.05, off - 0.07, len(tokens)), tokens))
        segments.append({"start": on, "end": off, "text": text, "words": None})
    ScriptedCtc(worker_module, audio, script).install(aligner)
    params = {"pad_sec": 0.4, "edge_sec": 0.12, "wide_sec": 3.0, "batch_sec": 20.0, "low_conf": 0.3, "max_inner_gap": 1.0}
    if refine is not None:
        params["refine"] = refine
    job = {"audio_npy": str(tmp_path / "audio.npy"), "segments": segments, "sample_rate": SR, "params": params}
    return job, aligner, speech


SPEC = [("あいうえおかき", ["あいう", "えおかき"], 1.0, 1.9, 0.06),
        ("さしすせそ", None, 3.0, 3.6, 0.08),
        ("たちつてと", ["たち", "つてと"], 4.5, 5.2, 0.04)]


def test_align_job_refines_words_and_segments_and_reports_the_check(worker_module, tmp_path, capsys):
    job, aligner, speech = make_scripted_job(worker_module, tmp_path, SPEC, {"enabled": True})
    result = worker_module.align_job(aligner, job)
    log = capsys.readouterr().out
    assert set(result) == {"spans", "confs", "wide", "word_spans", "checks"} and not any(result["wide"])

    for (text, words, on, off, cons), span in zip(SPEC, result["spans"]):
        assert abs(span[0] - on) <= 0.02 and abs(span[1] - off) <= 0.02       # 整句起訖貼到真正的聲音

    # 有拆詞的句子：詞的起訖也是；整句的起點 = 第一個詞的起點、終點 = 最後一個詞的終點
    for k in (0, 2):
        words = result["word_spans"][k]
        assert len(words) == 2 and words[0][1] < words[1][0] + 1e-9
        assert result["spans"][k] == [words[0][0], words[1][1]]
    assert len(result["word_spans"][1]) == 1                                  # 沒給詞的切法 → 整句一個詞
    assert result["word_spans"][1][0] == result["spans"][1]

    checks = result["checks"]
    assert [len(c) for c in checks] == [2, 1, 2]
    assert all(len(w) == 4 for c in checks for w in c)
    assert checks[0][0][0] == "moved" and checks[0][0][1] < -0.05             # 第一個詞的起點提早了 0.08 秒左右
    assert checks[0][1][2] == "moved" and checks[0][1][3] > 0.05              # 最後一個詞的終點延後了
    assert "[對齊驗證]" in log and "起點：" in log and "終點：" in log and "依聲音修正" in log


def test_align_job_without_refine_keeps_the_raw_ctc_times(worker_module, tmp_path):
    job, aligner, speech = make_scripted_job(worker_module, tmp_path, SPEC, None)
    result = worker_module.align_job(aligner, job)
    assert result["checks"] == [None, None, None]
    for (text, words, on, off, cons), span in zip(SPEC, result["spans"]):
        assert span[0] - on >= 0.06 and off - span[1] >= 0.04                 # 沒微調：起點晚、終點早，都差好幾十毫秒


def test_align_job_refine_disabled_flag_is_the_same_as_no_refine(worker_module, tmp_path):
    job, aligner, _ = make_scripted_job(worker_module, tmp_path, SPEC, {"enabled": False})
    assert worker_module.align_job(aligner, job)["checks"] == [None, None, None]


def test_align_job_refined_result_is_strictly_closer_to_the_truth(worker_module, tmp_path):
    raw_job, raw_aligner, speech = make_scripted_job(worker_module, tmp_path, SPEC, None)
    raw = worker_module.align_job(raw_aligner, raw_job)
    job, aligner, _ = make_scripted_job(worker_module, tmp_path, SPEC, {"enabled": True})
    refined = worker_module.align_job(aligner, job)
    for (on, off, _), a, b in zip(speech, raw["spans"], refined["spans"]):
        assert abs(b[0] - on) < abs(a[0] - on) and abs(b[1] - off) < abs(a[1] - off)


def test_align_job_skips_sentences_that_could_not_be_aligned(worker_module, tmp_path):
    spec = SPEC[:1] + [("。。。", None, 2.2, 2.6, 0.0)] + SPEC[1:2]
    job, aligner, _ = make_scripted_job(worker_module, tmp_path, spec, {"enabled": True})
    result = worker_module.align_job(aligner, job)
    assert result["spans"][1] is None and result["word_spans"][1] is None and result["checks"][1] is None
    assert result["checks"][0] is not None and result["checks"][2] is not None


def test_align_job_ctc_words_in_silence_are_flagged(worker_module, tmp_path):
    """Whisper 聽到了、CTC 也硬對上了，但那個位置根本沒有聲音（幻覺）→ 檢查結果是 silent，時間維持 CTC 的。"""
    job, aligner, _ = make_scripted_job(worker_module, tmp_path, SPEC, {"enabled": True},
                                        silent_spec=[("あいうえお", 7.0, 7.6)])
    result = worker_module.align_job(aligner, job)
    assert result["checks"][3] == [["silent", None, "silent", None]]
    assert result["spans"][3][0] == pytest.approx(7.05, abs=0.03)
    assert all(c[0][0] != "silent" for c in result["checks"][:3])


def test_a_weak_sound_between_loud_ones_is_not_reported_as_silence(worker_module):
    """實際使用的日誌裡，一段影片 373 條字幕有 74 條被標「聽不到聲音」，多半是「ク」「っ」「た」這種很短、很弱的詞：
    CTC 的一格（20 毫秒）剛好落在無聲子音、促音或兩個音節之間的小凹陷上。前後明明有大聲的字，不能算聽不到聲音。"""
    words = [(1.0, 1.4, 0.0), (1.42, 1.48, 0.06), (1.52, 1.9, 0.0), (4.0, 4.4, 0.0)]   # 中間那個詞只有 60 毫秒很弱的雜訊
    env = worker_module.energy_envelope_db(synth_speech(6, words), SR)
    new, checks = worker_module.refine_boundaries(env, [(1.05, 1.38), (1.44, 1.46), (1.58, 1.85), (4.05, 4.3)])
    assert "silent" not in (checks[1][0], checks[1][2])              # 弱音本身：可能貼得到、也可能無法判定，但不是「聽不到」
    assert all("silent" not in (c[0], c[2]) for c in checks)
    # 對照：詞的前後 0.1 秒都沒有聲音（落在兩段話中間的空白裡，附近 1 秒內有人在說話），才是聽不到
    words = [(1.0, 1.4, 0.0), (2.0, 2.4, 0.0), (3.4, 3.8, 0.0)]
    env = worker_module.energy_envelope_db(synth_speech(6, words), SR)
    new, checks = worker_module.refine_boundaries(env, [(1.05, 1.38), (2.05, 2.35), (2.9, 3.0), (3.45, 3.75)])
    assert checks[2] == ["silent", None, "silent", None] and new[2] == (2.9, 3.0)
    assert all("silent" not in (c[0], c[2]) for i, c in enumerate(checks) if i != 2)


def test_a_boundary_on_a_dip_is_unclear_not_silent(worker_module):
    """詞的前後 0.1 秒內有聲音（不算聽不到），但起訖點各落在空白上、離聲音超過搜尋範圍：無法判定（unclear），時間不動。"""
    env = np.full(600, -80.0)
    env[130:140] = -20.0                       # 0.66～0.71 秒有聲音
    env[300:340] = -20.0                       # 1.51～1.71 秒有聲音；中間是空白
    s, e, check = worker_module._refine_one(env, 0.76, 1.41, None, None, worker_module.REFINE_DEFAULTS)
    assert (s, e) == (0.76, 1.41) and check == ["unclear", None, "unclear", None]


def test_quiet_surroundings_are_silent_but_a_loud_background_is_only_noisy(worker_module):
    """低對比有兩種：附近整片都安靜（沒人在說話 → silent）、或背景太吵分不出來（noisy）。"""
    words = [(1.0, 1.5, 0.0), (1.7, 2.2, 0.0), (2.4, 2.9, 0.0), (4.0, 4.4, 0.0)]
    quiet = worker_module.energy_envelope_db(synth_speech(9, words), SR)
    # 第 4 個詞後面 5.5～7.5 秒是安靜的，CTC 卻在 6.5 秒放了一個詞
    new, checks = worker_module.refine_boundaries(quiet, [(1.05, 1.4), (1.75, 2.1), (2.45, 2.8), (4.05, 4.3), (6.5, 6.8)])
    assert checks[4] == ["silent", None, "silent", None] and new[4] == (6.5, 6.8)
    assert checks[0][0] == "moved"
    noisy = worker_module.energy_envelope_db(synth_speech(9, words, floor_db=-32.0), SR)
    new, checks = worker_module.refine_boundaries(noisy, [(1.05, 1.4), (6.5, 6.8)])
    assert checks[1] == ["noisy", None, "noisy", None]
