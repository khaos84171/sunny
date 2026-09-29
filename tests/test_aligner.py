"""AlignServer 與 run_alignment：透過真的 align_worker.py（配假模型）跑，涵蓋重用、逾時重啟、取消、出錯、通訊重試。"""
import os
import threading
import time

import numpy as np
import pytest

from helpers import TESTS, alive, wait_dead
from whisper_app import aligner, config
from whisper_app.jobs import JOBS, JobCancelled

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
