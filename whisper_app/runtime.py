"""啟動時要先做好的事:log 檔、輸出資料夾、啟動失敗的錯誤訊息框。只用標準函式庫，所以就算其他套件沒裝好也能把錯誤記下來、跳出訊息。"""

from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path

from . import config


# 啟動時發生、值得讓使用者知道的事（例如改用了備用資料夾），視窗開起來後寫進日誌
STARTUP_NOTES: list[str] = []


LOG_FILE = None          # 目前使用的 log 檔路徑；找不到可以寫入的位置時是 None
output_dir = ""          # 實際使用的輸出資料夾（setup_dirs() 之後才有值）
separation_work_dir = ""  # 實際使用的人聲分離暫存資料夾


class _RotatingLog(io.TextIOBase):
    """
    log 檔。sys.stdout / sys.stderr 都導向這裡；檔案超過 max_bytes 就換檔，只保留最近 backups 份舊的。
    多個執行緒同時寫不會交錯；壞掉的字元（例如檔名裡的孤立代理字元）寫成跳脫文字，不會讓寫入出錯。
    """

    def __init__(self, path, max_bytes: int, backups: int):
        self._path = Path(path)
        self._max_bytes = max_bytes
        self._backups = backups
        self._lock = threading.Lock()
        self._size = 0
        self._file = self._open()
        if self._size >= self._max_bytes:  # 上次留下來的已經太大
            self._rotate()

    def _open(self):
        f = open(self._path, "a", encoding="utf-8", errors="backslashreplace", buffering=1)
        self._size = os.path.getsize(self._path)
        return f

    def _rotate(self) -> None:
        self._file.close()
        try:
            for i in range(self._backups - 1, 0, -1):
                older = self._path.with_name(f"{self._path.name}.{i}")
                if older.exists():
                    os.replace(older, self._path.with_name(f"{self._path.name}.{i + 1}"))
            if self._backups > 0:
                os.replace(self._path, self._path.with_name(f"{self._path.name}.1"))
            else:
                self._path.unlink()
        except OSError:
            pass  # 檔案被別的程序占著（例如同時開了兩個視窗）換不了：繼續往原檔寫
        self._file = self._open()
        if self._size >= self._max_bytes:
            self._size = 0  # 換檔失敗：等再寫滿一輪才重試，不要每寫一行就試一次

    @property
    def encoding(self) -> str:
        return "utf-8"

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        with self._lock:
            self._file.write(s)
            self._size += len(s.encode("utf-8", errors="backslashreplace"))
            if self._size >= self._max_bytes:
                self._rotate()
        return len(s)

    def flush(self) -> None:
        with self._lock:
            if not self._file.closed:  # close() 之後 IOBase 還會再呼叫一次 flush
                self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()
        super().close()


def _open_log_file():
    """log 檔預設放在腳本旁邊；那個資料夾不能寫入（例如裝在 Program Files）就改放到暫存資料夾。"""
    for folder in (config.BASE_DIR, Path(tempfile.gettempdir())):
        path = folder / "whisper_app.log"
        try:
            return path, _RotatingLog(path, config.LOG_MAX_BYTES, config.LOG_BACKUPS)
        except OSError:
            continue
    return None, open(os.devnull, "w", encoding="utf-8")


def setup_logging() -> None:
    """
    把 sys.stdout / sys.stderr 導向 log 檔。用 pythonw 雙擊啟動時沒有終端機，sys.stdout / sys.stderr 會是 None，
    任何 print 或例外訊息原本會直接消失甚至報錯，所以先一律導向同一個 log 檔。最先呼叫。
    """
    global LOG_FILE
    LOG_FILE, stream = _open_log_file()
    sys.stdout = stream
    sys.stderr = stream
    if LOG_FILE is None:
        STARTUP_NOTES.append("!!! 找不到可以寫入的位置，這次不會留下 whisper_app.log")
    elif LOG_FILE.parent != config.BASE_DIR:
        STARTUP_NOTES.append(f"注意：腳本資料夾不能寫入，日誌檔改放在 {LOG_FILE}")
    logging.basicConfig()
    logging.getLogger("faster_whisper").setLevel(logging.DEBUG if config.DEBUG_LOG else logging.INFO)


def show_fatal_error(message: str):
    """啟動失敗或執行中發生未捕捉例外時，跳出訊息框告知使用者。"""
    traceback.print_exc()
    if os.environ.get("WHISPER_APP_NO_DIALOG"):  # 自動測試用：只記錄，不要跳出會卡住的訊息框
        return
    title = "Whisper 字幕產生器 - 發生錯誤"
    text = message + (f"\n\n詳細錯誤已寫入：\n{LOG_FILE}" if LOG_FILE else "")
    try:
        import tkinter as _tk
        from tkinter import messagebox as _mb
        _root = _tk.Tk()
        _root.withdraw()
        _mb.showerror(title, text)
        _root.destroy()
    except Exception:
        # tkinter 本身載入失敗（例如安裝 Python 時沒有勾 tcl/tk）：改用 Windows 內建的訊息框
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, text, title, 0x10)
        except Exception:
            pass


def _ensure_dir(preferred: str, name: str, label: str) -> str:
    """
    建立 preferred 資料夾並回傳實際可用的路徑。建不起來就依序改用腳本旁邊的 <name>、
    使用者家目錄下的 whisper_subtitles/<name>，並在 STARTUP_NOTES 記下改用了哪裡。
    """
    candidates = [Path(preferred), config.BASE_DIR / name,
                  Path(os.path.expanduser("~")) / "whisper_subtitles" / name]
    for i, folder in enumerate(candidates):
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"無法建立{label}資料夾 {folder}：{e}")
            continue
        if i > 0:
            STARTUP_NOTES.append(f"注意：{label}資料夾 {preferred} 建立不起來，改用 {folder}")
        return str(folder)
    raise OSError(f"{label}資料夾建立失敗：{preferred}（也試過 {candidates[1]}、{candidates[2]}）")


def setup_dirs() -> None:
    """建立輸出資料夾與人聲分離暫存資料夾（偏好的路徑建不起來會改用備用的）。失敗時跳訊息框並丟出例外。"""
    global output_dir, separation_work_dir
    try:
        output_dir = _ensure_dir(config.OUTPUT_DIR, "text", "輸出")
        separation_work_dir = _ensure_dir(config.SEPARATION_WORK_DIR, "separated", "人聲分離暫存")
    except Exception:
        show_fatal_error("無法建立輸出資料夾。請檢查 whisper_app/config.py 裡 OUTPUT_DIR／SEPARATION_WORK_DIR 的路徑。")
        raise
