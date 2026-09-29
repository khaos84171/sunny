"""pytest 共用設定。

- 把專案根目錄和 tests/stubs 放進 sys.path：測試不需要真的 faster-whisper、tkinterdnd2、GPU。
- 每個測試都自動隔離：設定檔、輸出資料夾指到暫存目錄，取消狀態、模型快取、常駐的對齊程序都在前後重設。
"""
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
for path in (str(TESTS / "stubs"), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    from whisper_app import aligner, config, devices, jobs, runtime, transcribe

    out, sep = tmp_path / "out", tmp_path / "sep"
    out.mkdir()
    sep.mkdir()
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(runtime, "output_dir", str(out), raising=False)
    monkeypatch.setattr(runtime, "separation_work_dir", str(sep), raising=False)
    monkeypatch.setattr(runtime, "STARTUP_NOTES", [])
    monkeypatch.setattr(devices, "_cuda_ok", None)
    monkeypatch.setattr(transcribe, "_loaded_model", None)
    yield
    jobs.JOBS.cancel()  # 停掉測試裡啟動的所有背景程序
    for proc in list(jobs.JOBS._procs):
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    jobs.JOBS._procs.clear()
    jobs.JOBS._generation = 0
    jobs.JOBS._job_generation = 0
    aligner.ALIGN_SERVER._proc = None


@pytest.fixture
def fake_demucs(monkeypatch):
    """讓 Demucs 指令找得到 tests/fakes 裡的假 demucs（同一個 Python 的 -m demucs.separate）。"""
    fakes = str(TESTS / "fakes")
    monkeypatch.syspath_prepend(fakes)
    monkeypatch.setenv("PYTHONPATH", fakes)
    import importlib
    importlib.invalidate_caches()
    return TESTS / "fakes"
