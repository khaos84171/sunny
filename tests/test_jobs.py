"""取消與背景程序管理：JobControl 登記、停止、取消計數；子程序環境。"""
import subprocess
import sys
import threading
import time

import pytest

from helpers import alive, subprocess_env, wait_dead
from whisper_app import jobs
from whisper_app.jobs import JOBS, JobCancelled

SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]


def test_initial_state_is_not_cancelled():
    assert not JOBS.is_cancelled()
    JOBS.check()                                           # 不會丟例外


def test_cancel_kills_registered_processes_and_marks_jobs_cancelled():
    proc = JOBS.spawn(SLEEPER)
    assert alive(proc.pid)
    generation = JOBS.generation
    JOBS.cancel()
    assert wait_dead(proc.pid) and JOBS.generation == generation + 1
    with pytest.raises(JobCancelled):
        JOBS.check()


def test_jobs_started_after_a_cancel_are_not_affected():
    old = JOBS.generation
    JOBS.cancel()
    assert JOBS.is_cancelled(old)                          # 取消前排進來的：作廢
    JOBS.begin(JOBS.generation)
    assert not JOBS.is_cancelled()                         # 取消之後新開始的：正常
    assert not JOBS.is_cancelled(JOBS.generation)


def test_process_spawned_while_cancelled_is_killed_immediately():
    """取消剛好發生在子程序啟動的空檔。"""
    JOBS.begin(JOBS.generation)
    JOBS._job_generation = JOBS.generation - 1 if JOBS.generation else -1     # 模擬：這個工作是取消之前排進來的
    JOBS._generation += 1
    proc = JOBS.spawn(SLEEPER)
    assert wait_dead(proc.pid)


def test_release_stops_a_running_process_and_unregisters_it():
    proc = JOBS.spawn(SLEEPER)
    pid = proc.pid
    JOBS.release(proc)
    assert wait_dead(pid) and proc not in JOBS._procs


def test_release_none_is_a_noop():
    JOBS.release(None)


def test_release_of_finished_process_only_unregisters_it():
    proc = JOBS.spawn([sys.executable, "-c", "pass"])
    proc.wait()
    JOBS.release(proc)
    assert proc not in JOBS._procs


def test_kill_tree_ignores_already_finished_process():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    jobs._kill_tree(proc)                                   # 不會出錯


def test_atexit_stops_registered_processes_when_the_program_ends(tmp_path):
    code = ("import sys\nfrom whisper_app.jobs import JOBS\n"
            "p = JOBS.spawn([sys.executable, '-c', 'import time; time.sleep(60)'])\nprint(p.pid, flush=True)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, env=subprocess_env())
    pid = int(out.stdout.strip().splitlines()[-1])
    assert wait_dead(pid, 5)


def test_child_env_forces_utf8_output():
    env = jobs._child_env(HF_TEST="1")
    assert env["PYTHONIOENCODING"] == "utf-8" and env["PYTHONUTF8"] == "1" and env["HF_TEST"] == "1"


def test_console_python_prefers_python_exe_next_to_pythonw(tmp_path, monkeypatch):
    (tmp_path / "python.exe").write_text("")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "pythonw.exe"))
    assert jobs._console_python() == str(tmp_path / "python.exe")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "other" / "pythonw.exe"))
    assert jobs._console_python().endswith("pythonw.exe")       # 旁邊沒有 python.exe 就照原樣
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
    assert jobs._console_python() == "/usr/bin/python3"


# ---------------- 可中斷的等待（Whisper 在主程式裡跑，殺不掉，只能「不等它」）----------------
def test_run_interruptibly_returns_the_result_and_passes_arguments():
    assert JOBS.run_interruptibly(lambda a, b=0: a + b, 1, b=2) == 3


def test_run_interruptibly_reraises_the_errors_of_the_call():
    def boom():
        raise ValueError("炸了")
    with pytest.raises(ValueError, match="炸了"):
        JOBS.run_interruptibly(boom)


def test_run_interruptibly_stops_waiting_as_soon_as_cancelled():
    release, finished = threading.Event(), threading.Event()

    def stuck():
        release.wait(10)                                   # 像 Whisper 在解碼：一旦開始就停不下來
        finished.set()
    JOBS.begin(JOBS.generation)
    threading.Timer(0.2, JOBS.cancel).start()
    t0 = time.time()
    try:
        with pytest.raises(JobCancelled):
            JOBS.run_interruptibly(stuck)
        assert time.time() - t0 < 1.5 and not finished.is_set()      # 呼叫本身還卡著，取消卻已經有反應
    finally:
        release.set()
    assert finished.wait(2)                                # 它在背景自己做完，不會出事


def test_run_interruptibly_does_not_even_start_when_already_cancelled():
    called = []
    JOBS.begin(JOBS.generation)
    JOBS.cancel()
    with pytest.raises(JobCancelled):
        JOBS.run_interruptibly(lambda: called.append(1))
    assert not called


def test_iter_interruptibly_yields_everything_in_order():
    assert list(JOBS.iter_interruptibly(iter(range(5)))) == [0, 1, 2, 3, 4]
    assert list(JOBS.iter_interruptibly([])) == []


def test_iter_interruptibly_reraises_the_errors_of_the_iterable():
    def gen():
        yield 1
        raise ValueError("炸了")
    it = JOBS.iter_interruptibly(gen())
    assert next(it) == 1
    with pytest.raises(ValueError, match="炸了"):
        next(it)


def test_iter_interruptibly_stops_waiting_when_cancelled_and_the_producer_only_finishes_its_current_item():
    release, closed, produced = threading.Event(), threading.Event(), []

    def gen():
        try:
            for i in range(10):
                if i == 1:
                    release.wait(10)                       # 第二項卡住（像 Whisper 在解碼一個 30 秒視窗）
                produced.append(i)
                yield i
        finally:
            closed.set()
    JOBS.begin(JOBS.generation)
    threading.Timer(0.3, JOBS.cancel).start()
    seen, t0 = [], time.time()
    try:
        with pytest.raises(JobCancelled):
            for item in JOBS.iter_interruptibly(gen()):
                seen.append(item)
        assert seen == [0] and time.time() - t0 < 1.5
    finally:
        release.set()
    assert closed.wait(2)                                  # 手上那一項做完就停、把 generator 收掉
    assert produced == [0, 1]                              # 不會再往下做第三項


def test_iter_interruptibly_stops_the_producer_when_the_consumer_leaves_early():
    produced = []

    def gen():
        for i in range(1000):
            produced.append(i)
            yield i
            time.sleep(0.01)
    it = JOBS.iter_interruptibly(gen())
    assert next(it) == 0
    it.close()
    time.sleep(0.3)
    count = len(produced)
    time.sleep(0.2)
    assert len(produced) == count < 1000                   # 呼叫端不要了，背景就不再往下做
