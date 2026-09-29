"""取消與背景程序管理：JobControl 登記、停止、取消計數；子程序環境。"""
import subprocess
import sys

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
