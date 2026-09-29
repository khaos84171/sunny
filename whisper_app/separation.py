"""用 Demucs 分離人聲（獨立程序、不跳黑視窗、進度接進 GUI、結果快取）。"""

from __future__ import annotations

import codecs
import collections
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .devices import resolve_device
from .jobs import JOBS, _NO_WINDOW, _child_env, _console_python


_PROGRESS_RE = re.compile(r"(\d{1,3})%\|")   # tqdm 進度條的樣子：「 45%|████▌     | …」


def _iter_console_lines(stream):
    """
    逐行讀子程序的輸出（bytes）。tqdm 進度條是用「回到行首」的方式原地更新的，
    所以回車字元也當作換行，不然整條進度會擠成一行、等到結束才讀得到。
    """
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    buf = ""
    while True:
        chunk = stream.read1(4096)
        if not chunk:
            break
        parts = re.split(r"[\r\n]+", buf + decoder.decode(chunk))
        buf = parts.pop()
        for part in parts:
            if part.strip():
                yield part
    buf += decoder.decode(b"", final=True)
    if buf.strip():
        yield buf


def _wav_is_complete(path: Path) -> bool:
    """
    WAV 檔頭記的音訊長度是不是真的都寫進檔案了。輸出到一半被中斷的檔案，
    檔頭記的長度會是 0 或比實際檔案大，這種不能當成已經分離好的結果。
    """
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return False
            pos = 12
            while True:
                chunk_head = f.read(8)
                if len(chunk_head) < 8:
                    return False
                chunk_id, chunk_size = chunk_head[:4], int.from_bytes(chunk_head[4:], "little")
                if chunk_id == b"data":
                    return chunk_size > 0 and pos + 8 + chunk_size <= size
                pos += 8 + chunk_size + (chunk_size & 1)  # 每個區塊都對齊到偶數位元組
                f.seek(pos)
    except OSError:
        return False


def _demucs_command() -> list[str]:
    """
    啟動 Demucs 的指令前半段。優先用「跑這個程式的同一個 Python」的 demucs（python -m demucs.separate），
    這樣用 .vbs 的 PYTHONW_PATH 指定虛擬環境時也找得到；那個環境沒裝 demucs 才退回 PATH 上的 demucs。
    """
    if importlib.util.find_spec("demucs") is not None:
        return [_console_python(), "-m", "demucs.separate"]
    exe = shutil.which("demucs")
    if exe:
        return [exe]
    raise FileNotFoundError("找不到 Demucs。請在執行這個程式的同一個 Python 環境裡執行 pip install demucs"
                            f"（目前用的 Python：{sys.executable}）")


def separate_vocals(input_path: str, work_dir: str, log_func,
                     device: str = "cuda", model_name: str = "htdemucs",
                     segment: int | None = None, progress_func=None) -> str:
    """
    用 Demucs 分離人聲與背景音樂，回傳人聲音軌路徑。分離很耗時，所以同一個檔案分離過就直接重用：
        結果放在 <work_dir>/<模型>/<檔名>__<來源檔大小與修改時間的雜湊>/vocals.wav，
        同名但內容不同的檔案（a.mp4 和 a.mkv、重新剪過的版本）不會共用到彼此的結果。
        Demucs 先輸出到暫存資料夾，確認檔案完整才搬進來，中途被中斷不會留下半個檔案被當成完成。
        舊版放在 <檔名>/vocals.wav 的結果，只要檔案完整、而且比來源檔新，也會沿用。
    progress_func(0～1) 回報分離進度（從 Demucs 輸出的進度條讀取），沒給就不回報。
    """
    input_path = Path(input_path)
    report = progress_func or (lambda frac: None)
    source = input_path.stat()
    requested_device, device = device, resolve_device(device)
    key = hashlib.sha1(f"{source.st_size}:{source.st_mtime_ns}".encode()).hexdigest()[:8]
    model_dir = Path(work_dir) / model_name
    final_dir = model_dir / f"{input_path.stem}__{key}"
    legacy_path = model_dir / input_path.stem / "vocals.wav"

    for cached, must_be_newer in ((final_dir / "vocals.wav", False), (legacy_path, True)):
        if (cached.exists() and _wav_is_complete(cached)
                and (not must_be_newer or cached.stat().st_mtime >= source.st_mtime)):
            log_func(f"[人聲分離] 已存在分離結果，跳過分離：{cached}")
            report(1.0)
            return str(cached)

    tmp_out = Path(work_dir) / f"_partial_{key}"
    shutil.rmtree(tmp_out, ignore_errors=True)  # 上次中斷留下的
    cmd = _demucs_command() + [
        "--two-stems", "vocals",
        "-n", model_name,
        "-d", device,
        "-o", str(tmp_out),
        str(input_path),
    ]
    if segment is not None:
        cmd += ["--segment", str(segment)]

    if device == "cpu" and requested_device == "auto":
        log_func("[人聲分離] 偵測不到可用的 NVIDIA GPU，改用 CPU 分離，會慢很多")
    log_func(f"[人聲分離] 開始處理：{input_path.name}（這步驟可能需要幾分鐘，視音訊長度與裝置而定）")
    print(f"[人聲分離] 指令：{' '.join(cmd)}")
    proc = None
    try:
        proc = JOBS.spawn(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=_child_env(), creationflags=_NO_WINDOW,
        )
        tail = collections.deque(maxlen=15)
        last_pct = -1
        for line in _iter_console_lines(proc.stdout):
            m = _PROGRESS_RE.search(line)
            if m:
                pct = min(int(m.group(1)), 100)
                if pct != last_pct:
                    last_pct = pct
                    report(pct / 100)
                continue
            line = line.strip()
            print(line)
            log_func("[人聲分離] " + line)
            tail.append(line)
        returncode = proc.wait()
        JOBS.check()  # 被取消時子程序是被停掉的，回傳碼不是 0，要先判斷這個
        if returncode != 0:
            detail = "\n".join(tail) if tail else "（沒有輸出）"
            if "not compiled with CUDA" in detail:
                detail += "\n（Demucs 用的 PyTorch 是 CPU 版：把 SEPARATION_DEVICE 改成 \"cpu\"，或安裝 CUDA 版 PyTorch）"
            raise RuntimeError(f"Demucs 人聲分離失敗（代碼 {returncode}）：\n{detail}")

        produced = next(iter((tmp_out / model_name).glob("*/vocals.wav")), None)
        if produced is None or not _wav_is_complete(produced):
            raise FileNotFoundError(f"Demucs 已結束，但找不到完整的人聲檔：{tmp_out / model_name}")
        shutil.rmtree(final_dir, ignore_errors=True)  # 前一次留下的不完整結果
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(produced.parent, final_dir)
    finally:
        JOBS.release(proc)
        shutil.rmtree(tmp_out, ignore_errors=True)

    report(1.0)
    log_func(f"[人聲分離] 完成，人聲音軌：{final_dir / 'vocals.wav'}")
    return str(final_dir / "vocals.wav")
