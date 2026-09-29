"""端到端：合成音訊（真正的起訖已知）→ 真的 align_worker 子程序（假模型照劇本放 CTC 的尖峰，帶著常見的偏差：
字幕起點落在母音、終點比聲音早結束）→ 套用結果 → 輸出 SRT → 用 evaluate 跟標準答案比。
驗證整條鏈路：聲音微調讓誤差大幅下降、交叉驗證的統計與「聽不到聲音」出現在日誌、關掉微調時回到 CTC 的誤差。

注意：這證明的是程式的串接與邏輯正確。真實音訊、真實模型上準確度到底多少，要用 evaluate 工具拿自己的影片量。"""
import json
import os
import random

import numpy as np
import pytest

from helpers import TESTS
from synth import SR, synth_speech
from whisper_app import aligner, config, evaluate, pipeline, runtime
from whisper_app.models import Word
from whisper_app.splitter import split_segment
from whisper_app.srt_io import write_srt
from whisper_app.timing import apply_alignment, finalize_aligned_timing

KANA = "あいうえおかきくけこさしすせそたちつてとなにぬねの"


def build_case(tmp_path, n=14, seed=3, ghost=False, offset=0.0):
    """
    寫出 ref.srt（標準答案）、rough.srt（粗略時間的輸入）、script.json、audio.npy，回傳音訊。
    ghost=True 時再加一條音訊裡沒有聲音、CTC 卻硬對上的字幕。
    offset：兩份 SRT 的時間整個加上這麼多秒（模擬剪輯軟體匯出、時間軸從 01:00:00 開始的字幕；音訊不變）。
    """
    rng = random.Random(seed)
    speech, spikes, ref, rough = [], [], [], []
    t = 1.0
    for _ in range(n):
        line_start = t
        text = ""
        for _ in range(rng.choice([1, 2, 3])):                       # 一句由 1～3 個詞組成，詞與詞之間有短停頓
            word = "".join(rng.choice(KANA) for _ in range(rng.randint(2, 4)))
            dur = 0.12 * len(word) + rng.uniform(0.05, 0.2)
            cons = rng.choice([0.0, 0.04, 0.06, 0.08])
            speech.append((t, t + dur, cons))
            start_ctc = t + cons + rng.uniform(0.015, 0.05)                              # CTC：落在母音
            end_ctc = max(t + dur - rng.uniform(0.05, 0.1), start_ctc + 0.04 * len(word))   # 終點比聲音早
            spikes += [[float(x), ch] for x, ch in zip(np.linspace(start_ctc, end_ctc - 0.02, len(word)), word)]
            text += word
            last_end = t + dur
            t += dur + rng.uniform(0.04, 0.08)
        ref.append({"start": line_start, "end": last_end, "text": text})
        # Whisper 給的粗略時間：前後各差 0.2 秒以內
        rough.append({"start": max(line_start + rng.uniform(-0.2, 0.2), 0.0), "end": last_end + rng.uniform(-0.2, 0.2), "text": text})
        t += rng.uniform(0.4, 0.9)
    total = t + 1.0
    if ghost:
        ghost_start = t + 0.5
        word = "あいうえお"
        spikes += [[float(x), ch] for x, ch in zip(np.linspace(ghost_start + 0.05, ghost_start + 0.55, len(word)), word)]
        ref.append({"start": ghost_start, "end": ghost_start + 0.7, "text": word})
        rough.append({"start": ghost_start, "end": ghost_start + 0.7, "text": word})
        total = ghost_start + 2.0
    audio = synth_speech(total, speech, seed=seed)
    np.save(tmp_path / "audio.npy", audio)
    (tmp_path / "script.json").write_text(json.dumps({"audio": str(tmp_path / "audio.npy"), "spikes": spikes}, ensure_ascii=False),
                                          encoding="utf-8")
    shift = lambda lines: [{**l, "start": l["start"] + offset, "end": l["end"] + offset} for l in lines]
    write_srt(str(tmp_path / "ref.srt"), shift(ref))
    write_srt(str(tmp_path / "rough.srt"), shift(rough))
    return audio


@pytest.fixture
def run_pipeline(monkeypatch, tmp_path):
    """用真的 align_worker（假模型照劇本放尖峰）對齊 rough.srt，回傳 (輸出 SRT 路徑, 日誌)。"""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(TESTS / "ml_stubs"), str(TESTS.parent)]))
    monkeypatch.setenv("FAKE_SCRIPT", str(tmp_path / "script.json"))
    monkeypatch.setattr(config, "ALIGN_KEEP_ALIVE_SEC", 0)          # 每個工作各開一個程序
    monkeypatch.setattr(config, "ALIGN_END_HOLD_SEC", 0.0)         # 拿掉「稍微延長」這種顯示上的習慣，輸出才跟真正的聲音直接比
    monkeypatch.setattr(config, "ALIGN_MIN_DURATION", 0.0)

    def run():
        audio = np.load(tmp_path / "audio.npy")
        monkeypatch.setattr(aligner, "decode_audio", lambda path, sampling_rate=16000: audio)
        logs = []
        out = pipeline.align_existing_srt(str(tmp_path / "rough.srt"), "/x/影片.mp4", False, logs.append, lambda p: None,
                                          add_blank=False)
        return out, logs
    return run


def test_sound_refinement_cuts_the_alignment_error_end_to_end(tmp_path, run_pipeline, monkeypatch):
    build_case(tmp_path)
    monkeypatch.setattr(config, "ALIGN_REFINE", False)
    ctc_out, _ = run_pipeline()
    monkeypatch.setattr(config, "ALIGN_REFINE", True)
    refined_out, logs = run_pipeline()
    assert ctc_out.endswith("_align_ctc.srt") and refined_out.endswith("_align.srt")

    ctc = evaluate.evaluate(tmp_path / "ref.srt", ctc_out)
    refined = evaluate.evaluate(tmp_path / "ref.srt", refined_out)
    assert ctc["matched"] == refined["matched"] == 14
    # 只用 CTC：起點晚（落在母音）、終點早（比聲音早結束），平均差好幾十毫秒，而且是有方向的系統性偏差
    assert ctc["start"]["mean_abs"] > 0.04 and ctc["start"]["bias"] > 0.03
    assert ctc["end"]["mean_abs"] > 0.05 and ctc["end"]["bias"] < -0.05
    assert ctc["start"]["within"][0.04] < 0.5
    # 聲音微調之後：平均誤差不到 20 毫秒，幾乎所有字幕都落在 40 毫秒內
    assert refined["start"]["mean_abs"] < 0.02 and refined["end"]["mean_abs"] < 0.02
    assert refined["start"]["within"][0.04] >= 0.95 and refined["end"]["within"][0.04] >= 0.95
    assert refined["start"]["mean_abs"] < ctc["start"]["mean_abs"] / 3 and refined["end"]["mean_abs"] < ctc["end"]["mean_abs"] / 3

    text = "\n".join(logs)
    assert "[對齊驗證]" in text and "依聲音修正" in text and "起點：" in text and "終點：" in text
    assert "建議檢查" not in text                                     # 每一條都有聲音、信心也夠：沒有可疑的字幕

    # 評估工具的交叉驗證也看得出來：CTC 版有固定的偏差可以修，微調版沒有
    assert ctc["cv_start"]["after"]["mean_abs"] < ctc["cv_start"]["before"]["mean_abs"] / 2
    assert refined["cv_start"]["before"]["mean_abs"] - refined["cv_start"]["after"]["mean_abs"] < 0.005


def test_subtitles_from_an_editor_timeline_starting_at_one_hour_are_aligned_and_stay_on_that_timeline(tmp_path, run_pipeline, monkeypatch):
    """剪輯軟體（DaVinci Resolve 等）匯出的字幕從 01:00:00 開始：整份差一小時，以前每一條都「對不上」。
    現在要自動偵測、扣掉再對齊，輸出仍然是那條時間軸，而且準確度跟沒有偏移時一樣。"""
    build_case(tmp_path, offset=3600.0)
    out, logs = run_pipeline()
    assert any("從 01:00:00,000 開始" in x for x in logs) and not any("對不上" in x for x in logs)
    assert not any("建議檢查" in x for x in logs)
    r = evaluate.evaluate(tmp_path / "ref.srt", out)
    assert r["matched"] == 14 and r["start"]["mean_abs"] < 0.02 and r["end"]["mean_abs"] < 0.02
    assert r["start"]["within"][0.04] >= 0.95
    first = evaluate.load_subtitles(out)[0]
    assert 3600 < first["start"] < 3610                                      # 仍然是 01:00:0x，不是被平移成 00:00:0x


def test_editor_timeline_with_leading_blank_starts_the_blank_at_the_timeline_origin(tmp_path, run_pipeline, monkeypatch):
    build_case(tmp_path, offset=3600.0)
    monkeypatch.setattr(aligner, "decode_audio", lambda path, sampling_rate=16000: np.load(tmp_path / "audio.npy"))
    logs = []
    out = pipeline.align_existing_srt(str(tmp_path / "rough.srt"), "/x/影片.mp4", False, logs.append, lambda p: None, add_blank=True)
    head = open(out, encoding="utf-8").read().split("\n\n")[0]
    assert head.startswith("1\n01:00:00,000 --> 01:00:0") and head.endswith("\u200b")


def test_subtitles_that_do_not_belong_to_the_video_are_rejected_with_an_explanation(tmp_path, run_pipeline):
    build_case(tmp_path, offset=5000.5)                                      # 差的不是整數小時：沒辦法自動處理
    with pytest.raises(ValueError, match="對不起來"):
        run_pipeline()


def test_a_line_with_no_sound_behind_it_is_reported(tmp_path, run_pipeline):
    """Whisper 聽到了、CTC 也硬對上，但音訊裡那個位置沒有聲音（幻覺）：日誌要點名，而且時間不亂動。"""
    build_case(tmp_path, ghost=True)
    out, logs = run_pipeline()
    text = "\n".join(logs)
    assert "建議檢查" in text and "聽不到聲音" in text and "あいうえお" in text
    reported = [l for l in logs if "位置聽不到聲音" in l]
    assert len(reported) == 1 and "#15" in reported[0]                 # 只有最後那條幻覺字幕
    assert any("起點：" in l and "該處聽不到聲音 1 個" in l for l in logs)  # 統計裡也有
    ref = evaluate.load_subtitles(tmp_path / "ref.srt")
    pred = evaluate.load_subtitles(out)
    assert abs(pred[-1]["start"] - ref[-1]["start"] - 0.05) < 0.03      # 保留 CTC 的時間（比 Whisper 給的晚 0.05 秒左右）


def test_split_subtitles_start_and_end_at_the_refined_word_boundaries(tmp_path, monkeypatch):
    """Whisper 把三個詞放在同一段：詞 2 與詞 3 之間有 0.7 秒的停頓。拆分要在停頓處切開，
    而且兩條字幕的起訖是「聲音微調後」的詞邊界（切點前後的邊界都貼回真正的聲音）。"""
    words = [("あいうえ", 1.00, 1.50, 0.06), ("おかき", 1.55, 2.00, 0.04), ("くけこさ", 2.70, 3.20, 0.08)]
    spikes = []
    for text, on, off, cons in words:
        start_ctc, end_ctc = on + cons + 0.03, off - 0.08             # 帶著常見的偏差
        spikes += [[float(x), ch] for x, ch in zip(np.linspace(start_ctc, end_ctc - 0.02, len(text)), text)]
    audio = synth_speech(4.5, [(on, off, cons) for _, on, off, cons in words])
    np.save(tmp_path / "audio.npy", audio)
    (tmp_path / "script.json").write_text(json.dumps({"audio": str(tmp_path / "audio.npy"), "spikes": spikes}, ensure_ascii=False),
                                          encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(TESTS / "ml_stubs"), str(TESTS.parent)]))
    monkeypatch.setenv("FAKE_SCRIPT", str(tmp_path / "script.json"))
    monkeypatch.setattr(config, "ALIGN_KEEP_ALIVE_SEC", 0)
    monkeypatch.setattr(config, "ALIGN_END_HOLD_SEC", 0.0)
    monkeypatch.setattr(config, "ALIGN_MIN_DURATION", 0.0)
    monkeypatch.setattr(aligner, "decode_audio", lambda path, sampling_rate=16000: audio)

    def run(refine):
        monkeypatch.setattr(config, "ALIGN_REFINE", refine)
        seg = {"start": 0.9, "end": 3.3, "text": "".join(w[0] for w in words),
               "words": [Word(on - 0.1, off + 0.1, text) for text, on, off, _ in words]}     # Whisper 給的詞時間：粗略
        result = aligner.run_alignment("/x/a.wav", [seg], lambda m: None, lambda p: None)
        apply_alignment([seg], result, lambda m: None)
        subs = split_segment(seg)
        finalize_aligned_timing([dict(s, aligned=True) for s in subs], result["duration"])
        return subs

    refined = run(True)
    assert [s["text"] for s in refined] == ["あいうえおかき", "くけこさ"]
    for sub, (on, off) in zip(refined, [(1.00, 2.00), (2.70, 3.20)]):
        assert abs(sub["start"] - on) <= 0.015 and abs(sub["end"] - off) <= 0.015
    raw = run(False)                                                   # 對照：只用 CTC 時，兩條都是晚開始、早結束
    for sub, (on, off) in zip(raw, [(1.00, 2.00), (2.70, 3.20)]):
        assert sub["start"] - on > 0.05 and off - sub["end"] > 0.05
