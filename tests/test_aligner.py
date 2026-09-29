"""AlignServer 與 run_alignment：透過真的 align_worker.py（配假模型）跑，涵蓋重用、逾時重啟、取消、出錯、通訊重試。"""
import json
import os
import threading
import time

import numpy as np
import pytest

from helpers import TESTS, alive, wait_dead
from whisper_app import aligner, config
from whisper_app.jobs import JOBS, JobCancelled
from whisper_app.models import Word

AUDIO = np.random.default_rng(5).normal(0, 0.1, 16000 * 40).astype(np.float32)
SEGMENTS = [{"start": 1.0 + 3 * i, "end": 3.0 + 3 * i, "text": t, "words": None} for i, t in enumerate(["今日は天気がいい", "熊本地震", "あいうえおかきく"])]


@pytest.fixture
def server(monkeypatch, tmp_path):
    """讓 run_alignment 用假模型的真 align_worker，並在測試結束時停掉常駐的對齊程序。"""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(TESTS / "ml_stubs"), str(TESTS.parent)]))
    monkeypatch.setenv("FAKE_MODEL_LOG", str(tmp_path / "model.log"))
    monkeypatch.setattr(aligner, "decode_audio", lambda path, sampling_rate=16000: AUDIO)
    monkeypatch.setattr(config, "ALIGN_KEEP_ALIVE_SEC", 60)

    class S:
        log = tmp_path / "model.log"
        loads = staticmethod(lambda: (tmp_path / "model.log").read_text().count("load ") if (tmp_path / "model.log").exists() else 0)

        @staticmethod
        def run(segments=SEGMENTS):
            logs, prog = [], []
            return aligner.run_alignment("/x/a.wav", segments, logs.append, prog.append), logs, prog

        @staticmethod
        def restart(keep_alive):
            """換設定前先停掉常駐的對齊程序，並等它真的結束。"""
            monkeypatch.setattr(config, "ALIGN_KEEP_ALIVE_SEC", keep_alive)
            proc = aligner.ALIGN_SERVER._proc
            JOBS.cancel()
            JOBS.begin(JOBS.generation)
            if proc is not None:
                proc.wait(5)
    yield S
    JOBS.cancel()


def test_consecutive_files_share_one_process_and_one_model_load(server, monkeypatch):
    spawned = []
    real_spawn = JOBS.spawn
    monkeypatch.setattr(JOBS, "spawn", lambda cmd, **kw: (spawned.append(cmd), real_spawn(cmd, **kw))[1])
    (r1, l1, p1), (r2, l2, p2), (r3, l3, _) = server.run(), server.run(), server.run()
    assert len(spawned) == 1 and server.loads() == 1
    assert r1["duration"] == 40.0 and len(r1["spans"]) == 3 and r1["spans"] == r2["spans"] == r3["spans"] and r1["confs"] == r3["confs"]
    assert any("啟動對齊程序" in x for x in l1) and not any("啟動對齊程序" in x for x in l2 + l3)
    assert any("沿用已載入" in x for x in l2) and any("沿用已載入" in x for x in l3)
    assert p1[-1] == 1.0 and p2[-1] == 1.0
    assert alive(aligner.ALIGN_SERVER._proc.pid) and aligner.ALIGN_SERVER._proc in JOBS._procs


def test_cancel_stops_the_idle_server_and_the_next_file_restarts_it(server):
    server.run()
    pid = aligner.ALIGN_SERVER._proc.pid
    JOBS.cancel()
    assert wait_dead(pid)
    JOBS.begin(JOBS.generation)
    result, logs, _ = server.run()
    assert aligner.ALIGN_SERVER._proc.pid != pid and len(result["spans"]) == 3 and server.loads() == 2
    assert any("啟動對齊程序" in x for x in logs)


def test_idle_timeout_ends_the_process_and_next_use_restarts_it(server):
    server.restart(1)
    server.run()
    pid = aligner.ALIGN_SERVER._proc.pid
    time.sleep(2.2)
    assert wait_dead(pid, 3)
    before = server.loads()
    result, logs, _ = server.run()
    assert len(result["spans"]) == 3 and server.loads() == before + 1 and any("啟動對齊程序" in x for x in logs)


def test_keep_alive_zero_reloads_the_model_for_every_file(server):
    server.restart(0)
    server.run()
    assert wait_dead(aligner.ALIGN_SERVER._proc.pid, 3)
    server.run()
    assert server.loads() == 2


def test_cancel_during_alignment_raises_jobcancelled_and_kills_the_process(server, monkeypatch):
    monkeypatch.setenv("FAKE_MODEL_DELAY", "0.4")
    server.restart(60)
    box, progress = {}, []

    def work():
        try:
            aligner.run_alignment("/x/a.wav", SEGMENTS, lambda m: None, progress.append)
        except BaseException as e:
            box["exc"] = e
    thread = threading.Thread(target=work)
    thread.start()
    end = time.time() + 20
    while not progress and time.time() < end:
        time.sleep(0.05)
    pid = aligner.ALIGN_SERVER._proc.pid
    assert alive(pid)
    t0 = time.time()
    JOBS.cancel()
    thread.join(15)
    assert isinstance(box.get("exc"), JobCancelled) and wait_dead(pid) and time.time() - t0 < 3
    monkeypatch.setenv("FAKE_MODEL_DELAY", "0")
    JOBS.begin(JOBS.generation)
    assert len(server.run()[0]["spans"]) == 3                       # 取消之後下一個檔案正常


def test_worker_error_raises_runtimeerror_and_next_file_recovers(server, monkeypatch):
    monkeypatch.setattr(config, "ALIGN_PAD_SEC", "不是數字")           # 讓 worker 讀參數時出錯
    with pytest.raises(RuntimeError, match="對齊程序失敗（代碼 1）") as e:
        server.run()
    assert "ValueError" in str(e.value)
    assert wait_dead(aligner.ALIGN_SERVER._proc.pid, 3)              # 出錯的 worker 自己結束，不留下狀態不明的程序
    monkeypatch.setattr(config, "ALIGN_PAD_SEC", 0.4)
    result, logs, _ = server.run()
    assert len(result["spans"]) == 3 and any("啟動對齊程序" in x for x in logs)


def capture_job(monkeypatch, checks=None):
    """不啟動真的 worker：記下送出去的 job.json，回一份固定的結果。"""
    sent = []

    def run(job_json, result_json, log, progress):
        job = json.loads(job_json.read_text(encoding="utf-8"))
        sent.append(job)
        n = len(job["segments"])
        result = {"spans": [[0.0, 1.0]] * n, "confs": [0.9] * n, "wide": [False] * n, "word_spans": [None] * n}
        if checks is not None:
            result["checks"] = checks
        result_json.write_text(json.dumps(result), encoding="utf-8")
        return 0, []
    monkeypatch.setattr(aligner.ALIGN_SERVER, "run", run)
    monkeypatch.setattr(aligner, "decode_audio", lambda path, sampling_rate=16000: AUDIO)
    return sent


def echo_worker(monkeypatch):
    """不啟動真的 worker：記下送出去的 job.json，結果是「每句縮進 0.1 秒」（詞的時間有一個 None），用來看時間軸有沒有換算對。"""
    sent = []

    def run(job_json, result_json, log, progress):
        job = json.loads(job_json.read_text(encoding="utf-8"))
        sent.append(job)
        segs = job["segments"]
        result = {"spans": [[s["start"] + 0.1, s["end"] - 0.1] for s in segs], "confs": [0.9] * len(segs), "wide": [False] * len(segs),
                  "word_spans": [[[s["start"] + 0.1, s["start"] + 0.5], None] for s in segs]}
        result_json.write_text(json.dumps(result), encoding="utf-8")
        return 0, []
    monkeypatch.setattr(aligner.ALIGN_SERVER, "run", run)
    monkeypatch.setattr(aligner, "decode_audio", lambda path, sampling_rate=16000: AUDIO)   # 40 秒
    return sent


def at(offset):
    return [{"start": offset + 1.0 + 3 * i, "end": offset + 3.0 + 3 * i, "text": t, "words": None}
            for i, t in enumerate(["今日は天気がいい", "熊本地震", "あいうえおかきく"])]


def test_timeline_offset_of_a_whole_hour_is_removed_before_aligning_and_added_back(monkeypatch):
    sent = echo_worker(monkeypatch)
    logs = []
    result = aligner.run_alignment("/x/a.wav", at(3600.0), logs.append, lambda p: None, detect_offset=True)
    assert [(s["start"], s["end"]) for s in sent[0]["segments"]] == [(1.0, 3.0), (4.0, 6.0), (7.0, 9.0)]      # worker 看到影片的時間軸
    assert result["offset"] == 3600.0 and result["duration"] == 3640.0
    assert result["spans"] == [[3601.1, 3602.9], [3604.1, 3605.9], [3607.1, 3608.9]]                            # 回來的是字幕原本的時間軸
    assert result["word_spans"][0] == [[3601.1, 3601.5], None]                                                    # None 維持 None
    assert any("01:00:00,000" in x and "提早 1 小時" in x for x in logs)


def test_no_offset_detection_by_default(monkeypatch):
    sent = echo_worker(monkeypatch)
    result = aligner.run_alignment("/x/a.wav", at(0.0), lambda m: None, lambda p: None)
    assert sent[0]["segments"][0]["start"] == 1.0 and result["offset"] == 0.0 and result["duration"] == 40.0
    sent = echo_worker(monkeypatch)                                                  # 轉錄流程的字幕時間本來就對，不做偵測
    aligner.run_alignment("/x/a.wav", at(3600.0), lambda m: None, lambda p: None)
    assert sent[0]["segments"][0]["start"] == 3601.0


def test_hopeless_timeline_raises_before_starting_the_worker(monkeypatch):
    sent = echo_worker(monkeypatch)
    with pytest.raises(ValueError, match="對不起來"):
        aligner.run_alignment("/x/a.wav", at(5000.5), lambda m: None, lambda p: None, detect_offset=True)
    assert sent == []


def test_lines_past_the_end_of_the_media_are_mentioned(monkeypatch):
    echo_worker(monkeypatch)
    lines = [{"start": 1.0 + i, "end": 1.5 + i, "text": "はい", "words": None} for i in range(20)]
    lines.append({"start": 100.0, "end": 101.0, "text": "遅い", "words": None})
    logs = []
    result = aligner.run_alignment("/x/a.wav", lines, logs.append, lambda p: None, detect_offset=True)
    assert result["offset"] == 0.0 and any("有 1 條字幕的起點超過影片長度" in x for x in logs)


def test_refine_settings_are_sent_to_the_worker(monkeypatch):
    sent = capture_job(monkeypatch)
    monkeypatch.setattr(config, "ALIGN_REFINE_BACK_SEC", 0.2)
    aligner.run_alignment("/x/a.wav", SEGMENTS, lambda m: None, lambda p: None)
    assert sent[0]["params"]["refine"] == {
        "enabled": True, "back_sec": 0.2, "fwd_sec": config.ALIGN_REFINE_FWD_SEC, "end_fwd_sec": config.ALIGN_REFINE_END_FWD_SEC,
        "end_back_sec": config.ALIGN_REFINE_END_BACK_SEC, "min_contrast_db": config.ALIGN_REFINE_MIN_CONTRAST_DB,
        "thr_frac": config.ALIGN_REFINE_THRESHOLD, "agree_sec": config.ALIGN_AGREE_SEC}
    monkeypatch.setattr(config, "ALIGN_REFINE", False)
    aligner.run_alignment("/x/a.wav", SEGMENTS, lambda m: None, lambda p: None)
    assert sent[1]["params"]["refine"]["enabled"] is False


def test_checks_from_the_worker_are_passed_on_and_default_to_none(monkeypatch):
    capture_job(monkeypatch)
    old = aligner.run_alignment("/x/a.wav", SEGMENTS, lambda m: None, lambda p: None)
    assert old["checks"] == [None, None, None]                                   # 舊版 worker 沒有這一項
    checks = [[["ok", 0.0, "moved", 0.1]], None, [["silent", None, "silent", None]]]
    capture_job(monkeypatch, checks)
    assert aligner.run_alignment("/x/a.wav", SEGMENTS, lambda m: None, lambda p: None)["checks"] == checks


def test_real_worker_returns_one_check_per_word_when_refining(server):
    two_words = [{**s, "words": [Word(s["start"], s["start"] + 1, s["text"][:2]), Word(s["start"] + 1, s["end"], s["text"][2:])]}
                 for s in SEGMENTS]
    result, logs, _ = server.run(two_words)
    assert len(result["checks"]) == 3
    assert all(c is None or (len(c) == 2 and all(len(w) == 4 for w in c)) for c in result["checks"])
    assert any("[對齊驗證]" in x for x in logs)                                    # 隨機雜訊音訊：全是無法判定，但有統計輸出


def test_real_worker_skips_the_check_when_refinement_is_off(server, monkeypatch):
    monkeypatch.setattr(config, "ALIGN_REFINE", False)
    result, logs, _ = server.run()
    assert result["checks"] == [None, None, None] and not any("[對齊驗證]" in x for x in logs)


def test_old_worker_without_serve_mode_gets_a_helpful_message(server, monkeypatch, tmp_path):
    old = tmp_path / "old_worker.py"
    old.write_text("import sys, json\nopen(sys.argv[1])\n", encoding="utf-8")          # 舊版：第一個參數當成 job.json 開檔
    monkeypatch.setattr(config, "ALIGN_WORKER", old)
    with pytest.raises(RuntimeError, match="一起更新"):
        server.run()


# ---------------- 通訊重試邏輯（假 worker）----------------
@pytest.fixture
def fake_worker(server, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ALIGN_WORKER", TESTS / "fakes" / "fake_serve_worker.py")
    monkeypatch.setenv("FAKE_COUNTER", str(tmp_path / "counter"))
    monkeypatch.setenv("PYTHONPATH", "")
    server.restart(60)
    return tmp_path / "counter"


def one(text):
    return [{"start": 0.0, "end": 1.0, "text": text, "words": None}]


def test_worker_that_ends_silently_before_starting_is_replaced_and_retried(server, fake_worker):
    result, logs, _ = server.run(one("die_silently"))
    assert len(result["spans"]) == 1 and int(fake_worker.read_text()) == 2 and any("第 2 個程序做完了" in x for x in logs)


def test_worker_that_disappears_midway_is_not_retried(server, fake_worker):
    with pytest.raises(RuntimeError, match="代碼 9"):
        server.run(one("crash"))
    assert int(fake_worker.read_text()) == 1


def test_worker_reporting_failure_gives_code_and_message(server, fake_worker):
    with pytest.raises(RuntimeError, match="代碼 2") as e:
        server.run(one("fail"))
    assert "boom" in str(e.value)
