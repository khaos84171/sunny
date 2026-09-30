"""常駐子程序（align_worker.py、asr_worker.py 的 --serve 模式）的主程式這一側:啟動、交工作、讀進度與訊息、重試。"""

from __future__ import annotations

import collections
import json
import subprocess
from pathlib import Path

from .jobs import JOBS, _NO_WINDOW, _child_env


class WorkerServer:
    """
    一個常駐的工作程序。載入模型要好幾秒，連續處理多個檔案時程序留著給下一個檔案用；
    閒置超過指定秒數它會自己結束並釋放記憶體。取消、關閉視窗時 JOBS 會把它一起停掉，下次要用再重新啟動。

    通訊協定（align_worker.py、asr_worker.py 都一樣）：工作一行一個寫進 stdin
    （JSON：{"job": job.json 路徑, "result": result.json 路徑}）；stdout 每行一則訊息：
    "START"、"END <0=成功>"、"PROGRESS <已完成> <總數>"、"LOG <文字>"、"ERROR <文字>"，其他行只寫進 log 檔。

    label：日誌前綴（例如「對齊」）；what：這個程序的稱呼（例如「對齊程序」）；
    command：回傳啟動指令的函式（每次啟動時才呼叫，改了設定下次重開就生效）。
    """

    def __init__(self, label: str, what: str, command):
        self.label = label
        self.what = what
        self._command = command
        self._proc = None

    def _ensure(self):
        """回傳可用的工作程序；還沒有或已經結束了就重新啟動。"""
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        JOBS.release(self._proc)  # 結束了的，從登記處拿掉
        self._proc = JOBS.spawn(
            self._command(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=_child_env(HF_HUB_DISABLE_SYMLINKS_WARNING="1"), creationflags=_NO_WINDOW,
        )
        return self._proc

    def _drop(self, proc) -> None:
        """丟掉一個確定不能用的程序（等它真的結束），下次 _ensure 一定開新的。
        剛結束的程序有一瞬間 poll() 還是 None，重試時不能再拿到它。"""
        JOBS.release(proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self._proc = None

    def run(self, job_json: Path, result_json: Path, log_func, progress_func):
        """交一個工作給程序並等它做完。回傳 (代碼, 最後幾行輸出)；代碼 0 = 成功。"""
        request = json.dumps({"job": str(job_json), "result": str(result_json)}, ensure_ascii=False) + "\n"
        for attempt in (1, 2):
            if self._proc is None or self._proc.poll() is not None:
                log_func(f"[{self.label}] 啟動{self.what}…")
            proc = self._ensure()
            try:
                proc.stdin.write(request)
                proc.stdin.flush()
            except OSError:  # 程序剛好在閒置逾時的瞬間結束了
                self._drop(proc)
                if attempt == 1:
                    continue
                raise
            started, code = False, None
            tail = collections.deque(maxlen=15)
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                if line.startswith("START"):
                    started = True
                elif line.startswith("END "):
                    code = int(line.split()[1])
                    break
                elif line.startswith("PROGRESS "):
                    done, total = line.split()[1:3]
                    progress_func(int(done) / max(int(total), 1))
                elif line.startswith("LOG "):
                    log_func(f"[{self.label}] " + line[4:])
                elif line.startswith("ERROR "):
                    log_func(f"!!! [{self.label}] " + line[6:])
                    tail.append(line[6:])
                elif line.strip():
                    print(line)  # transformers／NeMo 的警告、下載進度等只寫進 log 檔
                    tail.append(line)
            JOBS.check()  # 被取消時程序是被停掉的，輸出中斷，要先判斷這個
            if code is None:  # 程序在做完之前就沒了
                if not started and attempt == 1:
                    self._drop(proc)
                    continue  # 還沒開始做就結束了（剛好閒置逾時）：重開一個再試一次
                code = proc.wait() or 1
            return code, tail
