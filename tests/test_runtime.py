"""啟動流程：log 檔輪替、log 導向、輸出資料夾與備用位置、啟動失敗的訊息框。"""
import builtins
import logging
import os
import sys
import tempfile
import threading
import traceback
import types

import pytest

from whisper_app import config, runtime


# ---------------- 輪替的 log 檔 ----------------
def names(folder, prefix):
    return sorted(p.name for p in folder.glob(prefix + "*"))


def test_rotates_by_size_and_keeps_only_the_configured_backups(tmp_path):
    log = runtime._RotatingLog(tmp_path / "a.log", 1000, 3)
    for i in range(120):
        log.write(f"line {i:03d} " + "x" * 40 + "\n")
    log.flush()
    assert names(tmp_path, "a.log") == ["a.log", "a.log.1", "a.log.2", "a.log.3"]
    assert all((tmp_path / n).stat().st_size < 1100 for n in names(tmp_path, "a.log"))
    numbers = [int(line.split()[1]) for n in ("a.log.3", "a.log.2", "a.log.1", "a.log")
               for line in (tmp_path / n).read_text(encoding="utf-8").splitlines()]
    assert numbers == list(range(numbers[0], 120)) and numbers[0] > 0          # 舊→新連續、最新的在最後；只丟掉最舊的
    log.close()


def test_a_file_left_too_big_by_the_last_run_is_rotated_on_open(tmp_path):
    big = tmp_path / "b.log"
    big.write_text("z" * 5000, encoding="utf-8")
    log = runtime._RotatingLog(big, 1000, 2)
    assert (tmp_path / "b.log.1").exists() and big.stat().st_size == 0
    log.close()


def test_rotation_failure_keeps_writing_and_does_not_retry_on_every_line(tmp_path, monkeypatch):
    attempts = []

    def busy(*args, **kwargs):
        attempts.append(1)
        raise PermissionError("被別的程序占著")
    log = runtime._RotatingLog(tmp_path / "c.log", 200, 2)
    monkeypatch.setattr(runtime.os, "replace", busy)
    for _ in range(40):
        log.write("y" * 30 + "\n")
    monkeypatch.undo()
    log.flush()
    assert len(attempts) <= 12 and (tmp_path / "c.log").stat().st_size > 200
    log.close()


def test_concurrent_writers_do_not_lose_or_interleave_lines(tmp_path):
    log = runtime._RotatingLog(tmp_path / "t.log", 10 ** 9, 1)

    def spam(k):
        for i in range(500):
            log.write(f"thread {k} line {i}\n")
    threads = [threading.Thread(target=spam, args=(k,)) for k in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    log.close()
    lines = (tmp_path / "t.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4000 and all(l.startswith("thread ") and l[-1].isdigit() for l in lines)


def test_unencodable_characters_are_escaped_instead_of_raising(tmp_path):
    log = runtime._RotatingLog(tmp_path / "u.log", 10 ** 6, 1)
    log.write("正常中文 ✓ 孤立代理字元 \ud800 不會讓寫入出錯\n")
    log.close()
    text = (tmp_path / "u.log").read_text(encoding="utf-8")
    assert "正常中文 ✓" in text and "\\ud800" in text


def test_works_as_stdout_for_print_logging_and_tracebacks(tmp_path):
    log = runtime._RotatingLog(tmp_path / "v.log", 10 ** 6, 1)
    builtins.print("print 進來", file=log)
    logging.StreamHandler(log).handle(logging.LogRecord("t", logging.WARNING, __file__, 1, "logging 進來", None, None))
    try:
        1 / 0
    except ZeroDivisionError:
        traceback.print_exc(file=log)
    log.flush()
    text = (tmp_path / "v.log").read_text(encoding="utf-8")
    assert "print 進來" in text and "logging 進來" in text and "ZeroDivisionError" in text
    assert log.isatty() is False and log.encoding == "utf-8" and log.writable()


def test_close_then_flush_does_not_raise(tmp_path):
    """close() 之後 Python 結束時還會 flush 一次。"""
    log = runtime._RotatingLog(tmp_path / "w.log", 1000, 1)
    log.close()
    log.flush()


# ---------------- setup_logging ----------------
@pytest.fixture
def stdio(monkeypatch, tmp_path):
    """讓 setup_logging 改 sys.stdout/stderr 之後，測試結束會自動還原。"""
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    monkeypatch.setattr(runtime, "LOG_FILE", None)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    level = logging.getLogger("faster_whisper").level
    yield tmp_path
    logging.getLogger("faster_whisper").setLevel(level)
    if isinstance(sys.stdout, runtime._RotatingLog):
        sys.stdout.close()


def test_setup_logging_redirects_output_to_the_log_file(stdio):
    runtime.setup_logging()
    print("hello log")
    sys.stdout.flush()
    assert runtime.LOG_FILE == stdio / "whisper_app.log" and "hello log" in runtime.LOG_FILE.read_text(encoding="utf-8")
    assert sys.stderr is sys.stdout and runtime.STARTUP_NOTES == []


def test_log_falls_back_to_temp_folder_when_script_folder_is_not_writable(stdio):
    (stdio / "whisper_app.log").mkdir()                       # 同名資料夾擋住，開檔會失敗
    runtime.setup_logging()
    assert runtime.LOG_FILE == stdio / "tmp" / "whisper_app.log" and any("日誌檔改放在" in n for n in runtime.STARTUP_NOTES)


def test_log_is_discarded_when_nowhere_is_writable(stdio):
    (stdio / "whisper_app.log").mkdir()
    (stdio / "tmp" / "whisper_app.log").mkdir()
    runtime.setup_logging()
    print("still works")                                       # 不會崩潰
    assert runtime.LOG_FILE is None and any("不會留下 whisper_app.log" in n for n in runtime.STARTUP_NOTES)


@pytest.mark.parametrize("debug, level", [(False, logging.INFO), (True, logging.DEBUG)])
def test_faster_whisper_log_level_follows_debug_switch(stdio, monkeypatch, debug, level):
    monkeypatch.setattr(config, "DEBUG_LOG", debug)
    runtime.setup_logging()
    assert logging.getLogger("faster_whisper").level == level


# ---------------- 輸出資料夾 ----------------
@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_DIR", tmp_path / "app")
    (tmp_path / "app").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return tmp_path


def test_preferred_folder_is_used_when_it_can_be_created(home):
    assert runtime._ensure_dir(str(home / "pref" / "text"), "text", "輸出") == str(home / "pref" / "text")
    assert runtime.STARTUP_NOTES == [] and (home / "pref" / "text").is_dir()


def test_falls_back_to_the_script_folder_then_the_home_folder(home):
    blocker = home / "blocker"
    blocker.write_text("我是檔案不是資料夾")
    bad = str(blocker / "x" / "text")
    assert runtime._ensure_dir(bad, "text", "輸出") == str(home / "app" / "text")
    assert any("改用" in n for n in runtime.STARTUP_NOTES)
    (home / "app" / "text2").write_text("擋住")
    assert runtime._ensure_dir(bad, "text2", "輸出") == str(home / "home" / "whisper_subtitles" / "text2")


def test_error_when_nothing_can_be_created(home):
    blocker = home / "blocker"
    blocker.write_text("x")
    (home / "app" / "text").write_text("x")
    (home / "home" / "whisper_subtitles").write_text("x")
    with pytest.raises(OSError, match="建立失敗"):
        runtime._ensure_dir(str(blocker / "x"), "text", "輸出")


def test_setup_dirs_sets_the_resolved_folders(home, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(home / "o"))
    monkeypatch.setattr(config, "SEPARATION_WORK_DIR", str(home / "s"))
    runtime.setup_dirs()
    assert runtime.output_dir == str(home / "o") and runtime.separation_work_dir == str(home / "s")


def test_setup_dirs_failure_shows_the_error_and_reraises(home, monkeypatch):
    shown = []
    monkeypatch.setattr(runtime, "show_fatal_error", shown.append)
    blocker = home / "blocker"
    blocker.write_text("x")
    (home / "app" / "text").write_text("x")
    (home / "home" / "whisper_subtitles").write_text("x")
    monkeypatch.setattr(config, "OUTPUT_DIR", str(blocker / "x"))
    with pytest.raises(OSError):
        runtime.setup_dirs()
    assert shown and "config.py" in shown[0] and "OUTPUT_DIR" in shown[0]


# ---------------- 啟動失敗的訊息框 ----------------
def test_fatal_error_falls_back_to_windows_message_box_when_tkinter_is_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_APP_NO_DIALOG", raising=False)
    monkeypatch.setitem(sys.modules, "tkinter", None)                      # import tkinter 會失敗
    shown = []
    fake_ctypes = types.SimpleNamespace(windll=types.SimpleNamespace(user32=types.SimpleNamespace(
        MessageBoxW=lambda hwnd, text, title, flags: shown.append((text, title, flags)))))
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    monkeypatch.setattr(runtime, "LOG_FILE", tmp_path / "whisper_app.log")
    runtime.show_fatal_error("壞掉了")
    assert len(shown) == 1 and "壞掉了" in shown[0][0] and str(tmp_path / "whisper_app.log") in shown[0][0] and shown[0][2] == 0x10


def test_fatal_error_without_a_log_file_does_not_mention_a_path(monkeypatch):
    monkeypatch.delenv("WHISPER_APP_NO_DIALOG", raising=False)
    monkeypatch.setitem(sys.modules, "tkinter", None)
    shown = []
    monkeypatch.setitem(sys.modules, "ctypes", types.SimpleNamespace(windll=types.SimpleNamespace(
        user32=types.SimpleNamespace(MessageBoxW=lambda h, text, t, f: shown.append(text)))))
    monkeypatch.setattr(runtime, "LOG_FILE", None)
    runtime.show_fatal_error("壞掉了")
    assert shown == ["壞掉了"]


def test_fatal_error_never_raises_even_if_every_dialog_fails(monkeypatch):
    monkeypatch.delenv("WHISPER_APP_NO_DIALOG", raising=False)
    monkeypatch.setitem(sys.modules, "tkinter", None)
    monkeypatch.setitem(sys.modules, "ctypes", None)
    runtime.show_fatal_error("壞掉了")


def test_dialog_can_be_switched_off_for_automation(monkeypatch):
    monkeypatch.setenv("WHISPER_APP_NO_DIALOG", "1")
    monkeypatch.setitem(sys.modules, "tkinter", types.SimpleNamespace())    # 如果真的去用會爆
    runtime.show_fatal_error("不該跳出來")


# ---------------- 同時只開一個視窗 ----------------
@pytest.fixture
def lock_env(tmp_path, monkeypatch):
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    monkeypatch.setattr(runtime, "_instance_lock", None)
    yield tmp_path
    if runtime._instance_lock is not None:
        runtime._instance_lock.close()


def release(monkeypatch):
    runtime._instance_lock.close()
    monkeypatch.setattr(runtime, "_instance_lock", None)


def test_only_one_instance_at_a_time_and_lock_can_be_taken_again_after_release(lock_env, monkeypatch):
    assert runtime.acquire_single_instance() is True
    assert runtime.acquire_single_instance() is False                      # 第二個被擋下
    release(monkeypatch)
    assert runtime.acquire_single_instance() is True                       # 放掉之後又可以了


def test_multiple_instances_allowed_when_configured(lock_env, monkeypatch):
    monkeypatch.setattr(config, "ALLOW_MULTIPLE_INSTANCES", True)
    assert runtime.acquire_single_instance() is True and runtime.acquire_single_instance() is True
    assert not (lock_env / "whisper_app.lock").exists()


def test_separate_installations_do_not_block_each_other(lock_env, monkeypatch, tmp_path):
    assert runtime.acquire_single_instance() is True
    other = tmp_path / "另一份"
    other.mkdir()
    monkeypatch.setattr(config, "BASE_DIR", other)
    assert runtime.acquire_single_instance() is True


def test_falls_back_to_a_lock_in_the_temp_folder_and_still_excludes(lock_env, monkeypatch):
    (lock_env / "whisper_app.lock").mkdir()                                # 同名資料夾擋住，程式資料夾裡建不起鎖檔
    assert runtime.acquire_single_instance() is True
    assert runtime.acquire_single_instance() is False
    assert list((lock_env / "tmp").glob("whisper_app_*.lock"))


def test_never_blocks_the_user_when_no_lock_file_can_be_created(lock_env):
    (lock_env / "whisper_app.lock").mkdir()
    import hashlib
    tag = hashlib.sha1(str(lock_env).encode("utf-8")).hexdigest()[:8]
    (lock_env / "tmp" / f"whisper_app_{tag}.lock").mkdir()
    assert runtime.acquire_single_instance() is True and runtime.acquire_single_instance() is True


@pytest.mark.parametrize("already_locked, expected", [(False, True), (True, False)])
def test_windows_uses_msvcrt_locking(lock_env, monkeypatch, already_locked, expected):
    calls = []

    def locking(fd, mode, nbytes):
        calls.append((mode, nbytes))
        if already_locked:
            raise PermissionError("Permission denied")
    monkeypatch.setitem(sys.modules, "msvcrt", types.SimpleNamespace(LK_NBLCK=2, locking=locking))
    fake_os = types.SimpleNamespace(**{k: getattr(os, k) for k in dir(os) if not k.startswith("__")})
    fake_os.name = "nt"
    monkeypatch.setattr(runtime, "os", fake_os)
    assert runtime.acquire_single_instance() is expected
    assert calls == [(2, 1)]                                               # LK_NBLCK（不等待）、鎖 1 個位元組


# ---------------- 一般提示 / Python 提示 ----------------
def test_notice_is_logged_and_shown_as_an_information_box(monkeypatch, capsys):
    monkeypatch.delenv("WHISPER_APP_NO_DIALOG", raising=False)
    shown = []
    fake_tk = types.SimpleNamespace(Tk=lambda: types.SimpleNamespace(withdraw=lambda: None, destroy=lambda: None))
    fake_box = types.SimpleNamespace(showinfo=lambda title, text: shown.append(("info", text)),
                                     showerror=lambda title, text: shown.append(("error", text)))
    monkeypatch.setitem(sys.modules, "tkinter", types.SimpleNamespace(Tk=fake_tk.Tk, messagebox=fake_box))
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", fake_box)
    runtime.show_notice("已經開著了")
    assert shown == [("info", "已經開著了")] and "已經開著了" in capsys.readouterr().out


def test_notice_falls_back_to_windows_information_box(monkeypatch):
    monkeypatch.delenv("WHISPER_APP_NO_DIALOG", raising=False)
    monkeypatch.setitem(sys.modules, "tkinter", None)
    shown = []
    monkeypatch.setitem(sys.modules, "ctypes", types.SimpleNamespace(windll=types.SimpleNamespace(
        user32=types.SimpleNamespace(MessageBoxW=lambda hwnd, text, title, flags: shown.append((text, flags))))))
    runtime.show_notice("已經開著了")
    assert shown == [("已經開著了", 0x40)]                                    # 0x40 = 資訊圖示（錯誤是 0x10）


def test_fatal_error_message_is_also_written_to_the_log(monkeypatch, capsys):
    monkeypatch.setenv("WHISPER_APP_NO_DIALOG", "1")
    runtime.show_fatal_error("缺少必要套件")
    assert "缺少必要套件" in capsys.readouterr().out


def test_python_hint_points_at_the_console_python_next_to_pythonw(tmp_path, monkeypatch):
    (tmp_path / "python.exe").write_text("")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "pythonw.exe"))
    hint = runtime.python_hint()
    assert str(tmp_path / "pythonw.exe") in hint and f'"{tmp_path / "python.exe"}" -m pip install' in hint and "tkinterdnd2" in hint
    monkeypatch.setattr(sys, "executable", str(tmp_path / "elsewhere" / "pythonw.exe"))
    assert "elsewhere" in runtime.python_hint().split("-m pip")[0]        # 旁邊沒有 python.exe 就照原樣
