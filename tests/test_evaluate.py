"""評估工具（whisper_app/evaluate.py）：用人工校正過的 SRT 量對齊誤差，含交叉驗證。"""
import json
import random
import subprocess
import sys

import pytest

from helpers import ROOT
from whisper_app import config, evaluate
from whisper_app.srt_io import write_srt


def make_pair(tmp_path, start_bias=0.06, end_bias=-0.05, n=40, noise=0.01, seed=1, drop=(), rename=None):
    """標準答案 + 一份「起點偏 start_bias、終點偏 end_bias」的輸出。"""
    rng = random.Random(seed)
    ref, pred, t = [], [], 1.0
    for i in range(n):
        d = rng.uniform(1.0, 2.5)
        text = f"字幕{i}番のテキスト{i * 7}"
        ref.append({"start": t, "end": t + d, "text": text})
        if i not in drop:
            pred.append({"start": t + start_bias + rng.gauss(0, noise), "end": t + d + end_bias + rng.gauss(0, noise),
                         "text": (rename or {}).get(i, text)})
        t += d + rng.uniform(0.1, 0.8)
    write_srt(str(tmp_path / "ref.srt"), ref)
    write_srt(str(tmp_path / "pred.srt"), pred)
    return tmp_path / "ref.srt", tmp_path / "pred.srt"


# ---------------- 文字與讀檔 ----------------
def test_normalize_text_ignores_punctuation_spaces_case_and_width():
    assert evaluate.normalize_text("Ｑｕｉｚ Knock、はい！ 「OK」\n") == "quizknockはいok"
    assert evaluate.normalize_text("。、！ ") == ""


def test_load_subtitles_drops_blank_ones_and_sorts(tmp_path):
    write_srt(str(tmp_path / "a.srt"), [
        {"start": 5.0, "end": 6.0, "text": "後"}, {"start": 0.0, "end": 1.0, "text": "​"},   # 開頭空白字幕
        {"start": 1.0, "end": 2.0, "text": "。。。"}, {"start": 2.0, "end": 3.0, "text": "前"}])
    subs = evaluate.load_subtitles(tmp_path / "a.srt")
    assert [s["text"] for s in subs] == ["前", "後"]                       # 空白與只有標點的字幕不列入


# ---------------- 配對 ----------------
def subs(*items):
    return [{"start": s, "end": s + 1, "text": t} for s, t in items]


def test_matching_pairs_the_same_text_in_order():
    ref = subs((1, "こんにちは"), (3, "さようなら"), (5, "ありがとう"))
    pred = subs((1.1, "こんにちは。"), (3.2, "さようなら"), (5.1, "ありがとう"))
    assert [(i, j) for i, j, _ in evaluate.match_subtitles(ref, pred)] == [(0, 0), (1, 1), (2, 2)]


def test_matching_skips_lines_that_are_missing_or_reworded():
    ref = subs((1, "こんにちは"), (3, "さようなら"), (5, "ありがとう"))
    pred = subs((1.1, "こんにちは"), (5.1, "ありがとう"), (9, "無関係"))
    assert [(i, j) for i, j, _ in evaluate.match_subtitles(ref, pred)] == [(0, 0), (2, 1)]
    reworded = subs((1.1, "こんばんは"), (3.1, "さようなら"))                # 「こんにちは」→「こんばんは」：相似度不夠
    assert [(i, j) for i, j, _ in evaluate.match_subtitles(ref, reworded)] == [(1, 1)]
    assert [(i, j) for i, j, _ in evaluate.match_subtitles(ref, reworded, min_similarity=0.5)] == [(0, 0), (1, 1)]


def test_matching_does_not_pair_a_split_subtitle_with_half_of_it():
    """輸出把一條拆成兩條：只有一半的文字，不能拿它的終點去跟整條的終點比。"""
    ref = subs((1, "今日はいい天気ですねこれから出かけます"))
    pred = [{"start": 1.0, "end": 2.0, "text": "今日はいい天気ですね"}, {"start": 2.0, "end": 3.0, "text": "これから出かけます"}]
    assert evaluate.match_subtitles(ref, pred) == []


def test_matching_prefers_the_closest_in_time_among_identical_texts():
    ref = subs((10, "はい"))
    pred = subs((3, "はい"), (9.8, "はい"), (16, "はい"))
    assert [(i, j) for i, j, _ in evaluate.match_subtitles(ref, pred, max_shift=8)] == [(0, 1)]


def test_matching_ignores_candidates_too_far_in_time_and_never_goes_backwards():
    ref = subs((10, "はい"), (20, "はい"))
    pred = subs((1, "はい"), (20.2, "はい"))
    pairs = evaluate.match_subtitles(ref, pred, max_shift=3)
    assert [(i, j) for i, j, _ in pairs] == [(1, 1)]                        # 第一條的候選（1 秒）離太遠
    ref2 = subs((10, "AAA"), (12, "BBB"))
    pred2 = subs((10.1, "BBB"), (12.1, "AAA"))                               # 順序倒過來：配了 AAA 之後，BBB 已經在它前面，不能再配
    assert [(i, j) for i, j, _ in evaluate.match_subtitles(ref2, pred2, max_shift=3)] == [(0, 1)]


# ---------------- 統計 ----------------
def test_summarize_numbers():
    s = evaluate.summarize([0.02, -0.04, 0.10, 0.06], (0.04, 0.08, 0.12))
    assert s["n"] == 4 and s["mean_abs"] == pytest.approx(0.055) and s["max"] == pytest.approx(0.10)
    assert s["median_abs"] == pytest.approx(0.05) and s["bias"] == pytest.approx(0.04)   # 有號中位數 (0.02+0.06)/2
    assert s["within"] == {0.04: 0.5, 0.08: 0.75, 0.12: 1.0}


def test_summarize_of_nothing():
    assert evaluate.summarize([]) == {"n": 0}


def test_percentile_interpolates():
    assert evaluate._percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert evaluate._percentile([], 0.9) == 0.0 and evaluate._percentile([7.0], 0.9) == 7.0


# ---------------- 交叉驗證 ----------------
def test_cross_validation_removes_a_constant_offset_on_unseen_data():
    rng = random.Random(0)
    errors = [0.07 + rng.gauss(0, 0.01) for _ in range(50)]
    cv = evaluate.cross_validate_offset(errors, folds=5)
    assert cv["folds"] == 5 and cv["before"]["mean_abs"] > 0.06 and cv["after"]["mean_abs"] < 0.015
    assert all(o == pytest.approx(0.07, abs=0.01) for o in cv["offsets"])


def test_cross_validation_shows_no_gain_when_there_is_no_constant_offset():
    rng = random.Random(1)
    errors = [rng.gauss(0, 0.05) for _ in range(60)]
    cv = evaluate.cross_validate_offset(errors, folds=5)
    assert cv["before"]["mean_abs"] - cv["after"]["mean_abs"] < 0.003


def test_cross_validation_is_honest_about_offsets_that_do_not_generalise():
    """前半偏晚 0.1、後半偏早 0.1：用另一半算出來的偏移量拿來修正這一半，只會更糟——不能因為整體中位數而說有幫助。"""
    cv = evaluate.cross_validate_offset([0.1] * 10 + [-0.1] * 10, folds=2)
    assert cv["after"]["mean_abs"] == pytest.approx(0.2) and cv["before"]["mean_abs"] == pytest.approx(0.1)


def test_cross_validation_folds_are_contiguous_and_the_offset_comes_only_from_the_other_folds():
    errors = [0.0] * 5 + [1.0] * 5 + [2.0] * 5
    cv = evaluate.cross_validate_offset(errors, folds=3)
    # 留下第 1 折（全 0）→ 用其餘 10 筆（5 個 1、5 個 2）的中位數 1.5；依此類推
    assert cv["offsets"] == pytest.approx([1.5, 1.0, 0.5])


def test_cross_validation_needs_enough_data():
    assert evaluate.cross_validate_offset([0.1, 0.2, 0.3], folds=5) is None
    assert evaluate.cross_validate_offset([], folds=5) is None
    assert evaluate.cross_validate_offset([0.1] * 4, folds=5)["folds"] == 2      # 資料少時自動減少折數


# ---------------- 整份比較 ----------------
def test_evaluate_measures_the_known_offsets(tmp_path):
    ref, pred = make_pair(tmp_path, start_bias=0.06, end_bias=-0.05)
    r = evaluate.evaluate(ref, pred)
    assert r["ref_count"] == r["pred_count"] == r["matched"] == 40
    assert r["start"]["bias"] == pytest.approx(0.06, abs=0.01) and r["end"]["bias"] == pytest.approx(-0.05, abs=0.01)
    assert r["start"]["within"][0.04] < 0.2 and r["start"]["within"][0.12] > 0.95
    assert r["cv_start"]["after"]["mean_abs"] < 0.015 and r["cv_end"]["after"]["mean_abs"] < 0.015
    size = [max(abs(w["start_error"]), abs(w["end_error"])) for w in r["worst"]]
    assert len(size) == 40 and size == sorted(size, reverse=True) and r["worst"][0]["text"]     # 由大到小排好


def test_evaluate_reports_unmatched_lines(tmp_path):
    ref, pred = make_pair(tmp_path, drop=(3, 4), rename={7: "全然違う文章です"})
    r = evaluate.evaluate(ref, pred)
    assert r["ref_count"] == 40 and r["pred_count"] == 38 and r["matched"] == 37


def test_evaluate_with_no_common_subtitles(tmp_path):
    write_srt(str(tmp_path / "a.srt"), [{"start": 1, "end": 2, "text": "こんにちは"}])
    write_srt(str(tmp_path / "b.srt"), [{"start": 1, "end": 2, "text": "さようなら"}])
    r = evaluate.evaluate(tmp_path / "a.srt", tmp_path / "b.srt")
    assert r["matched"] == 0 and r["start"] == {"n": 0}
    assert "沒有任何一條配對成功" in evaluate.format_report(r)


# ---------------- 報告 ----------------
def test_report_suggests_new_settings_based_on_the_current_ones(tmp_path, monkeypatch):
    ref, pred = make_pair(tmp_path, start_bias=0.06, end_bias=-0.05)
    monkeypatch.setattr(config, "ALIGN_START_LEAD_SEC", 0.1)
    monkeypatch.setattr(config, "ALIGN_END_HOLD_SEC", 0.25)
    text = evaluate.format_report(evaluate.evaluate(ref, pred))
    assert "ALIGN_START_LEAD_SEC 從 0.1 改成 0.16" in text        # 輸出偏晚 0.06 → 要再提早 0.06
    assert "ALIGN_END_HOLD_SEC 從 0.25 改成 0.30" in text         # 終點偏早 0.05 → 多留 0.05
    assert "輸出偏晚" in text and "輸出偏早" in text and "配對成功 40 條" in text


def test_report_says_a_constant_offset_would_not_help_when_there_is_none(tmp_path):
    ref, pred = make_pair(tmp_path, start_bias=0.0, end_bias=0.0, noise=0.03)
    text = evaluate.format_report(evaluate.evaluate(ref, pred))
    assert "固定偏移量沒有明顯幫助" in text and "建議" not in text


def test_report_lists_the_worst_lines(tmp_path):
    ref, pred = make_pair(tmp_path)
    text = evaluate.format_report(evaluate.evaluate(ref, pred), show=3)
    assert "誤差最大的 3 條" in text
    assert evaluate.format_report(evaluate.evaluate(ref, pred), show=0).count("誤差最大") == 0


def test_comparison_table_has_one_row_per_output(tmp_path):
    ref, a = make_pair(tmp_path, start_bias=0.08, end_bias=-0.06)
    a.rename(tmp_path / "ctc.srt")
    ref, b = make_pair(tmp_path, start_bias=0.0, end_bias=0.0)
    b.rename(tmp_path / "refined.srt")
    results = [evaluate.evaluate(ref, tmp_path / "ctc.srt"), evaluate.evaluate(ref, tmp_path / "refined.srt")]
    table = evaluate.format_comparison(results)
    assert "ctc.srt" in table and "refined.srt" in table
    assert results[1]["start"]["mean_abs"] < results[0]["start"]["mean_abs"] / 3


# ---------------- 命令列 ----------------
def test_cli_prints_reports_and_saves_json(tmp_path, capsys):
    ref, pred = make_pair(tmp_path)
    out = tmp_path / "結果.json"
    assert evaluate.main([str(ref), str(pred), str(pred), "--json", str(out), "--show", "1", "--tolerance", "0.1", "0.05"]) == 0
    text = capsys.readouterr().out
    assert text.count("對照標準答案") == 2 and "總表" in text and "<=0.05秒  <=0.1秒" in text
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert len(saved) == 2 and saved[0]["matched"] == 40


def test_cli_reports_missing_files(tmp_path, capsys):
    ref, pred = make_pair(tmp_path)
    assert evaluate.main([str(ref), str(tmp_path / "沒有.srt")]) == 2
    assert "找不到檔案" in capsys.readouterr().err


def test_module_can_be_run_from_the_command_line(tmp_path):
    ref, pred = make_pair(tmp_path)
    done = subprocess.run([sys.executable, "-m", "whisper_app.evaluate", str(ref), str(pred)], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    assert done.returncode == 0 and "配對成功 40 條" in done.stdout and "起點" in done.stdout
