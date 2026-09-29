"""取消與背景程序的管理:JobControl 登記所有子程序（Demucs、對齊），按「取消處理」、關閉視窗、程式結束時一起停掉。"""

from __future__ import annotations

import atexit
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


CANCEL_POLL_SEC = 0.1   # 等一個停不下來的工作時，每隔多久看一次有沒有被取消（也就是取消最慢的反應時間）


class JobCancelled(Exception):
    """使用者按了「取消處理」（或關閉視窗），這個檔案不用做了。"""


def _kill_tree(proc) -> None:
    """
    停掉子程序。Windows 上用 taskkill /T 連它底下的子程序一起停（pip 裝的 demucs.exe 只是個啟動器，
    真正在跑的是它開出來的 python.exe）；其他系統直接 kill。已經結束的不做事。
    """
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, creationflags=_NO_WINDOW, timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


class JobControl:
    """
    「取消」的狀態，以及背景子程序（Demucs、對齊）的登記處。

    按一次取消 generation 就加 1：在那之前排進佇列的、正在跑的工作全部作廢。用計數而不是旗標，
    取消剛好發生在工作開始的瞬間也不會漏掉。子程序都要用 spawn() 啟動，取消、關閉視窗、
    程式結束時才能一起停掉，不會留在背景占著 GPU。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._generation = 0
        self._job_generation = 0   # 目前這個工作是在哪個 generation 排進來的
        self._procs = set()

    @property
    def generation(self) -> int:
        return self._generation

    def begin(self, generation: int) -> None:
        """工作開始：記下它是在哪個 generation 排進來的。"""
        self._job_generation = generation

    def is_cancelled(self, generation: int | None = None) -> bool:
        return self._generation != (self._job_generation if generation is None else generation)

    def check(self) -> None:
        """目前的工作被取消了就丟出 JobCancelled。各處理階段之間呼叫。"""
        if self.is_cancelled():
            raise JobCancelled()

    def cancel(self) -> None:
        with self._lock:
            self._generation += 1
            procs = list(self._procs)
        for proc in procs:
            _kill_tree(proc)

    def run_interruptibly(self, fn, *args, **kwargs):
        """
        跑一個一旦開始就停不下來的呼叫（載入 Whisper 模型、解碼整段音訊…它們在主程式裡面跑，沒辦法像子程序一樣殺掉）。
        工作交給背景執行緒，這裡每 CANCEL_POLL_SEC 秒看一次有沒有被取消：取消了就馬上丟出 JobCancelled，
        不用等它做完；它自己在背景跑完就算了，結果直接丟掉。沒被取消就回傳 fn 的結果，fn 丟的例外也照樣丟出來。
        """
        self.check()  # 已經取消了就不用開始
        box = queue.Queue(maxsize=1)

        def target():
            try:
                box.put((True, fn(*args, **kwargs)))
            except BaseException as e:  # 連同例外一起交給等的那一邊
                box.put((False, e))

        threading.Thread(target=target, name="interruptible-call", daemon=True).start()
        while True:
            try:
                ok, value = box.get(timeout=CANCEL_POLL_SEC)
            except queue.Empty:
                self.check()
                continue
            if ok:
                return value
            raise value

    def iter_interruptibly(self, iterable):
        """
        逐項取出 iterable（例如 Whisper 一段一段吐出來的轉錄結果），做法跟 run_interruptibly 一樣：
        產生下一項的過程在背景執行緒裡跑，這裡等的時候一直看有沒有被取消。
        被取消（或呼叫端提早離開）時，背景那一邊做完手上這一項就不再往下做，並把 iterable 收掉。
        """
        items = queue.Queue()
        stop = threading.Event()
        done = object()

        def pump():
            it = iter(iterable)
            try:
                while not stop.is_set():
                    try:
                        item = next(it)
                    except StopIteration:
                        items.put((done, None))
                        return
                    items.put((True, item))
            except BaseException as e:
                items.put((False, e))
            finally:
                close = getattr(it, "close", None)  # generator 只能由執行它的這個執行緒收掉
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass

        threading.Thread(target=pump, name="interruptible-iter", daemon=True).start()
        try:
            while True:
                try:
                    tag, value = items.get(timeout=CANCEL_POLL_SEC)
                except queue.Empty:
                    self.check()
                    continue
                if tag is done:
                    return
                if not tag:
                    raise value
                yield value
        finally:
            stop.set()

    def spawn(self, cmd, **kwargs):
        proc = subprocess.Popen(cmd, **kwargs)
        with self._lock:
            self._procs.add(proc)
        if self.is_cancelled():  # 取消剛好發生在啟動的空檔
            _kill_tree(proc)
        return proc

    def release(self, proc) -> None:
        """子程序用完了：還在跑的就停掉，並從登記處拿掉。"""
        if proc is None:
            return
        _kill_tree(proc)
        with self._lock:
            self._procs.discard(proc)


JOBS = JobControl()


atexit.register(JOBS.cancel)  # 不管怎麼結束，都不要留下背景程序


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # Windows：不要為子程序跳出黑色 cmd 視窗


def _child_env(**extra) -> dict:
    """子程序的環境變數：輸出一律用 UTF-8，主程式才不會讀到亂碼。"""
    return dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", **extra)


def _console_python() -> str:
    """用 pythonw 雙擊啟動時，子程序改用同資料夾的 python.exe，stdout 才能正常傳回進度。"""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        candidate = exe.with_name("python.exe")
        if candidate.exists():
            return str(candidate)
    return str(exe)
