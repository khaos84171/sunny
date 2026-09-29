"""align_worker.py（獨立程序）：黃金檔（結果不變）、常駐模式、顯存搬移、閒置逾時、出錯、萬用字元、frame 時間換算。用假的 torch/transformers。
聲音能量微調與交叉驗證另外在 test_align_refine.py。"""
import importlib
import json
import sys

import numpy as np
import pytest

from helpers import ROOT, TESTS, make_align_job, run_worker


def request(job, result):
    return json.dumps({"job": str(job), "result": str(result)})


def assert_same_result(actual: dict, golden: dict) -> None:
    """比對對齊結果：結構（哪些是 null、哪些用了第二輪）要完全一樣，數字容許極小的浮點誤差。"""
    assert actual.keys() == golden.keys() and actual["wide"] == golden["wide"]
    for key in ("spans", "confs", "word_spans"):
        np.testing.assert_allclose(_flatten(actual[key]), _flatten(golden[key]), rtol=1e-6, atol=1e-6, err_msg=key)


def _flatten(value):
    """把巢狀的 list（None 當作 -1）攤平成數字陣列，方便用容許誤差比對。"""
    if value is None:
        return [-1.0]
    if isinstance(value, (list, tuple)):
        return [x for v in value for x in _flatten(v)]
    return [float(value)]


# ---------------- 結果不變（黃金檔）----------------
@pytest.mark.parametrize("seed", [11, 13])
def test_matches_the_golden_output(tmp_path, seed):
    job = make_align_job(tmp_path / "job", seed=seed)
    golden = json.loads((TESTS / "golden" / f"align_seed{seed}_result.json").read_text(encoding="utf-8"))
    golden_stdout = (TESTS / "golden" / f"align_seed{seed}_stdout.txt").read_text(encoding="utf-8")
    code, out, _ = run_worker([job, tmp_path / "r.json"])
    assert code == 0 and out.replace("\r\n", "\n") == golden_stdout
    assert_same_result(json.loads((tmp_path / "r.json").read_text(encoding="utf-8")), golden)


@pytest.mark.parametrize("seed", [11, 13])
def test_only_difference_from_the_old_frame_time_estimate_is_a_small_shift(tmp_path, seed):
    """
    golden/legacy/ 是修正 frame 時間換算之前的輸出。換算改成精確的 0.02 秒之後，
    哪些句子對得上、哪一輪、信心都必須完全一樣，只有時間差一點點：最多半格（10 毫秒），而且只會變早。
    """
    job = make_align_job(tmp_path / "job", seed=seed)
    legacy = json.loads((TESTS / "golden" / "legacy" / f"align_seed{seed}_result.json").read_text(encoding="utf-8"))
    code, _, _ = run_worker([job, tmp_path / "r.json"])
    new = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert code == 0 and new["wide"] == legacy["wide"] and new["confs"] == legacy["confs"]
    for key in ("spans", "word_spans"):
        a, b = np.array(_flatten(new[key])), np.array(_flatten(legacy[key]))
        assert a.shape == b.shape and ((a == -1) == (b == -1)).all()          # 哪些是 null 完全一樣
        diff = (a - b)[a != -1]
        assert diff.max() <= 1e-9 and diff.min() >= -0.0101 and diff.min() < -0.001


def test_serve_mode_gives_the_same_results_as_one_shot_mode(tmp_path):
    job1, job2 = make_align_job(tmp_path / "a", seed=11), make_align_job(tmp_path / "b", seed=13)
    code, out, _ = run_worker(["--serve", 30], [request(job1, tmp_path / "r1.json"), request(job2, tmp_path / "r2.json")])
    assert code == 0 and out.count("START") == 2 and out.count("END 0") == 2
    for n, seed in ((1, 11), (2, 13)):
        golden = json.loads((TESTS / "golden" / f"align_seed{seed}_result.json").read_text(encoding="utf-8"))
        assert_same_result(json.loads((tmp_path / f"r{n}.json").read_text(encoding="utf-8")), golden)


def test_model_is_loaded_only_once_in_serve_mode(tmp_path):
    job = make_align_job(tmp_path / "j")
    log = tmp_path / "model.log"
    code, out, _ = run_worker(["--serve", 30], [request(job, tmp_path / "r1.json"), request(job, tmp_path / "r2.json")],
                              {"FAKE_MODEL_LOG": log})
    assert code == 0 and log.read_text().count("load ") == 1
    assert out.count("對齊模型載入完成") == 1 and out.count("沿用已載入的對齊模型") == 1


# ---------------- 顯存搬移 ----------------
def test_model_is_parked_on_cpu_between_jobs_when_on_gpu(tmp_path):
    job = make_align_job(tmp_path / "j", device="cuda")
    log = tmp_path / "model.log"
    run_worker(["--serve", 30], [request(job, tmp_path / "r1.json"), request(job, tmp_path / "r2.json")], {"FAKE_CUDA": 1, "FAKE_MODEL_LOG": log})
    assert log.read_text().split() == ["load", "fake/wav2vec2", "to", "cuda", "to", "cpu", "empty_cache", "to", "cuda", "to", "cpu", "empty_cache"]


def test_no_moving_around_without_a_gpu(tmp_path):
    job = make_align_job(tmp_path / "j", device="cuda")
    log = tmp_path / "model.log"
    _, out, _ = run_worker(["--serve", 30], [request(job, tmp_path / "r.json")], {"FAKE_CUDA": 0, "FAKE_MODEL_LOG": log})
    assert log.read_text().split() == ["load", "fake/wav2vec2", "to", "cpu"] and "改用 CPU" in out


@pytest.mark.parametrize("gpu, expect_cuda", [(0, False), (1, True)])
def test_device_auto_picks_gpu_only_when_present(tmp_path, gpu, expect_cuda):
    job = make_align_job(tmp_path / "j", device="auto")
    log = tmp_path / "model.log"
    _, out, _ = run_worker(["--serve", 30], [request(job, tmp_path / "r.json")], {"FAKE_CUDA": gpu, "FAKE_MODEL_LOG": log})
    assert ("to cuda" in log.read_text()) is expect_cuda and ("改用 CPU" in out) is (not expect_cuda)


# ---------------- 生命週期 ----------------
def test_exits_by_itself_after_the_idle_timeout(tmp_path):
    job = make_align_job(tmp_path / "j")
    code, out, seconds = run_worker(["--serve", 1], [request(job, tmp_path / "r.json")], close_stdin=False)
    assert code == 0 and "END 0" in out and "閒置超過 1 秒" in out and seconds < 20


def test_serve_zero_stops_after_one_job(tmp_path):
    job = make_align_job(tmp_path / "j")
    code, out, _ = run_worker(["--serve", 0], [request(job, tmp_path / "r1.json"), request(job, tmp_path / "r2.json")], close_stdin=False)
    assert code == 0 and out.count("END 0") == 1 and (tmp_path / "r1.json").exists() and not (tmp_path / "r2.json").exists()


def test_closing_stdin_ends_the_worker_cleanly():
    code, out, _ = run_worker(["--serve", 30])
    assert code == 0 and "START" not in out


def test_error_reports_and_ends_the_process(tmp_path):
    job = make_align_job(tmp_path / "j")
    code, out, _ = run_worker(["--serve", 30], [request(tmp_path / "不存在.json", tmp_path / "x.json"), request(job, tmp_path / "r.json")])
    assert code == 1 and "ERROR FileNotFoundError" in out and out.rstrip().endswith("END 1") and not (tmp_path / "r.json").exists()


def test_missing_audio_file_is_reported(tmp_path):
    job_path = make_align_job(tmp_path / "j")
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["audio_npy"] = str(tmp_path / "沒有這個.npy")
    job_path.write_text(json.dumps(job), encoding="utf-8")
    code, out, _ = run_worker(["--serve", 30], [request(job_path, tmp_path / "r.json")])
    assert code == 1 and "ERROR FileNotFoundError" in out and out.rstrip().endswith("END 1")


def test_one_shot_command_line_still_works(tmp_path):
    job = make_align_job(tmp_path / "j", seed=11)
    code, out, _ = run_worker([job, tmp_path / "once.json"])
    assert code == 0 and "START" not in out and (tmp_path / "once.json").exists()


# ---------------- 萬用字元 ----------------
def real_char_columns(aligner):
    special = {"<pad>", "<s>", "</s>", "<unk>", "|"}
    return [v for k, v in aligner.vocab.items() if k not in special]


@pytest.mark.parametrize("special_hot", ["0", "1"])
def test_wildcard_column_ignores_special_tokens(worker_module, monkeypatch, special_hot):
    """萬用字元「任何發音都可以」是真的字裡最高機率再扣分；<s> </s> <unk> | 就算機率異常高也不能被它吸走。"""
    monkeypatch.setenv("FAKE_SPECIAL_HOT", special_hot)
    aligner = worker_module.Aligner("x", "cpu")
    chunk = np.random.default_rng(3).normal(0, 0.1, 16000 * 2).astype(np.float32)
    lp, seconds = aligner.log_probs(chunk, 16000)
    real = lp[:, real_char_columns(aligner)].max(axis=1)
    np.testing.assert_allclose(lp[:, aligner.star_id], real - worker_module.STAR_PENALTY)
    assert seconds == pytest.approx(0.02, rel=0.05)


# ---------------- frame 時間換算 ----------------
@pytest.mark.parametrize("samples", [24000, 24100, 33333, 160007, 16000 * 26])
def test_frame_length_is_the_exact_model_stride(worker_module, samples):
    """第 t 個 frame 從第 t * 320 個取樣點開始 = t * 0.02 秒，跟視窗長度無關。
    以前用「視窗秒數 / frame 數」估，越靠近視窗尾端越晚，最多晚一整格（20 毫秒）。"""
    aligner = worker_module.Aligner("x", "cpu")
    assert aligner.frame_stride == 320
    lp, seconds = aligner.log_probs(np.zeros(samples, dtype=np.float32), 16000)
    assert seconds == 0.02


def test_frame_length_with_the_frame_count_of_the_real_wav2vec2(worker_module):
    """真的 wav2vec2 的 frame 數是 (取樣點 - 400) // 320 + 1（假模型是 取樣點 // 320），兩種都要算得出 0.02。"""
    aligner = worker_module.Aligner("x", "cpu")
    samples = 16000 * 5
    frames = (samples - 400) // 320 + 1
    aligner._posteriors = lambda chunk, sr: np.full((frames, len(aligner.vocab)), -5.0)
    _, seconds = aligner.log_probs(np.zeros(samples, dtype=np.float32), 16000)
    assert seconds == 0.02 and (samples / 16000) / frames > 0.02          # 舊的估法會多估


def test_frame_length_falls_back_to_an_estimate_when_the_stride_is_unknown_or_inconsistent(worker_module):
    aligner = worker_module.Aligner("x", "cpu")
    samples = 24100
    lp, _ = aligner.log_probs(np.zeros(samples, dtype=np.float32), 16000)
    estimate = (samples / 16000) / lp.shape[0]
    aligner.frame_stride = None                                         # 取不到模型步幅
    assert aligner.log_probs(np.zeros(samples, dtype=np.float32), 16000)[1] == pytest.approx(estimate)
    aligner.frame_stride = 640                                          # 步幅跟實際 frame 數對不上（不是這種模型）
    assert aligner.log_probs(np.zeros(samples, dtype=np.float32), 16000)[1] == pytest.approx(estimate)


def test_missing_or_odd_model_config_does_not_break_loading(worker_module, monkeypatch):
    from types import SimpleNamespace

    import transformers                       # tests/ml_stubs 裡假的（worker_module 已把它放進搜尋路徑）
    for config in (None, SimpleNamespace(), SimpleNamespace(conv_stride=None), SimpleNamespace(conv_stride="abc")):
        monkeypatch.setattr(transformers.Wav2Vec2ForCTC, "config", config, raising=False)
        assert worker_module.Aligner("x", "cpu").frame_stride is None


def test_wildcard_candidates_exclude_blank_delimiter_and_special_tokens(worker_module):
    aligner = worker_module.Aligner("x", "cpu")
    assert set(aligner.star_candidates.tolist()).isdisjoint({0, 1, 2, 3, 4})
    assert len(aligner.star_candidates) == len(aligner.vocab) - 5


# ---------------- 文字 → token、CTC ----------------
def test_text_to_tokens_maps_kana_swaps_and_wildcards(worker_module):
    vocab = {"<pad>": 0, "|": 1, "あ": 2, "い": 3, "う": 4, "く": 5, "ま": 6, "も": 7, "と": 8, "ー": 9}
    unknown = set()
    tokens = worker_module.text_to_tokens("くまもと ABC 123 ア。ー", vocab, 1, 99, unknown)
    assert tokens == [5, 6, 7, 8, 99, 99, 99, 99, 2, 9]         # ABC → 2 個萬用字元、123 → 3 個…；ア 換成 あ；標點略過
    assert unknown == {"a", "b", "c", "1", "2", "3"}


@pytest.mark.parametrize("length", [40, 63, 64, 250])
def test_ctc_alignment_recovers_a_clean_path_for_any_length(worker_module, length):
    """token 數 >= 64 時，NumPy 2 下舊寫法會 OverflowError（int 減 int8）。"""
    rng = np.random.default_rng(0)
    frames, vocab_size = 1300, 2500
    tokens = list(rng.integers(1, vocab_size, size=length))
    lp = np.log(np.full((frames, vocab_size), 1e-4))
    positions = np.linspace(20, frames - 20, length).astype(int)
    lp[:, 0] = np.log(0.9)
    for t, tok in zip(positions, tokens):
        lp[t, :] = np.log(1e-4)
        lp[t, tok] = np.log(0.95)
    first, end, best = worker_module.ctc_align(lp, tokens, 0)
    assert np.abs(first - positions).max() == 0


def test_ctc_alignment_returns_none_when_audio_is_too_short(worker_module):
    lp = np.log(np.full((5, 10), 0.1))
    assert worker_module.ctc_align(lp, [1, 2, 3, 4, 5, 6, 7], 0) is None
    assert worker_module.ctc_align(lp, [], 0) is None
