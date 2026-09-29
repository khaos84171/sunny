"""啟動流程：log 檔輪替、log 導向、輸出資料夾與備用位置、啟動失敗的訊息框。"""
import builtins
import logging
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
