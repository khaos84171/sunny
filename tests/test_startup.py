"""入口 w1_1.py 的啟動保護：用 pythonw 雙擊時沒有終端機，啟動失敗一定要留下線索。
在暫存資料夾裡放一份入口與套件的複本來跑，不會弄髒專案，也不會建立你設定的 D 槽資料夾。"""
import os
import re
import shutil
import subprocess
import sys

import pytest

from helpers import ROOT, TESTS


@pytest.fixture
def app_copy(tmp_path):
    dst = tmp_path / "app"
    shutil.copytree(ROOT / "whisper_app", dst / "whisper_app", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "w1_1.py", dst / "w1_1.py")
    config = dst / "whisper_app" / "config.py"
    text = config.read_text(encoding="utf-8")
    text = re.sub(r'^OUTPUT_DIR = .*$', f'OUTPUT_DIR = r"{tmp_path / "out" / "text"}"', text, flags=re.M)
    text = re.sub(r'^SEPARATION_WORK_DIR = .*$', f'SEPARATION_WORK_DIR = r"{tmp_path / "out" / "separated"}"', text, flags=re.M)
    config.write_text(text, encoding="utf-8")
    return dst


def run_entry(app_dir, code=None, pythonpath=(), timeout=60):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, pythonpath)), WHISPER_APP_NO_DIALOG="1", PYTHONIOENCODING="utf-8")
    args = [sys.executable, "-c", code] if code else [sys.executable, str(app_dir / "w1_1.py")]
    return subprocess.run(args, cwd=app_dir, env=env, capture_output=True, text=True, timeout=timeout)


def test_missing_package_folder_leaves_an_explanation_file(app_copy):
    shutil.rmtree(app_copy / "whisper_app")
    result = run_entry(app_copy)
    assert result.returncode != 0
    note = (app_copy / "whisper_startup_error.txt").read_text(encoding="utf-8")
    assert "whisper_app" in note and "ModuleNotFoundError" in note


def test_missing_required_package_is_written_to_the_log_file(app_copy):
    """必要套件（這裡模擬 tkinterdnd2）沒裝好：錯誤要進 log 檔，不能無聲消失。"""
    pytest.importorskip("tkinter")           # 沒有 tkinter 的話，失敗的原因會變成 tkinter，就不是這個測試要驗證的了
    result = run_entry(app_copy, pythonpath=[TESTS / "broken", TESTS / "stubs"])
    assert result.returncode != 0
    log = (app_copy / "whisper_app.log").read_text(encoding="utf-8")
    assert "ImportError" in log and "simulated" in log and "Traceback" in log


def test_normal_start_sets_up_logging_and_folders_without_opening_a_window(app_copy, tmp_path):
    pytest.importorskip("tkinter")
    code = ("import sys\nimport w1_1\nprint('loaded', callable(w1_1.main), w1_1.runtime.LOG_FILE.name, file=sys.__stdout__)\n"
            "print('寫進 log 的一行')\n")
    result = run_entry(app_copy, code, pythonpath=[TESTS / "stubs"])
    assert result.returncode == 0, result.stderr
    assert "loaded True whisper_app.log" in result.stdout
    assert "寫進 log 的一行" in (app_copy / "whisper_app.log").read_text(encoding="utf-8")
    assert (tmp_path / "out" / "text").is_dir() and (tmp_path / "out" / "separated").is_dir()
