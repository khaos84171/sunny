"""Demucs 人聲分離：呼叫方式、進度、快取（含被截斷的殘檔）、取消、找 Demucs 的順序。用 tests/fakes 裡的假 demucs。"""
import hashlib
import json
import os
import threading
import time
import wave
from pathlib import Path

import pytest

from helpers import alive, wait_dead, wait_for
from whisper_app import separation
from whisper_app.jobs import JOBS, JobCancelled


def make_wav(path: Path, seconds: float = 1.0, rate: int = 8000, extra_chunk: bool = False) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(rate * seconds))
    if extra_chunk:      # 在 fmt 和 data 之間插入一個 LIST 區塊
        b = path.read_bytes()
        i = b.index(b"data")
        path.write_bytes(b[:i] + b"LIST" + (5).to_bytes(4, "little") + b"abcde" + b"\x00" + b[i:])


class Drip:
    """一次只給一個位元組的串流（測試中文被切開、\\r 進度條）。"""
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def read1(self, n):
        if self.pos >= len(self.data):
            return b""
        self.pos += 1
        return self.data[self.pos - 1:self.pos]


@pytest.fixture
def env(fake_demucs, tmp_path, monkeypatch):
    calls = tmp_path / "calls.log"
    monkeypatch.setenv("FAKE_LOG", str(calls))
    monkeypatch.setenv("FAKE_MODE", "ok")
    src = tmp_path / "測試影片 第1話.mp4"
    src.write_bytes(b"x" * 999)
    work = tmp_path / "work"
    work.mkdir()

    class E:
        pass
    e = E()
    e.src, e.work, e.calls, e.tmp = src, work, calls, tmp_path
    e.count = lambda: len(calls.read_text(encoding="utf-8").splitlines()) if calls.exists() else 0
    e.argv = lambda: json.loads(calls.read_text(encoding="utf-8").splitlines()[-1])

    def run(path=None, device="cuda", **kw):
        logs, prog = [], []
        result = separation.separate_vocals(str(path or src), str(work), logs.append, device=device, model_name="htdemucs",
                                            segment=7, progress_func=prog.append, **kw)
        return result, logs, prog
    e.run = run
    return e


# ---------------- WAV 完整性 ----------------
def test_wav_is_complete_accepts_good_files(tmp_path):
    make_wav(tmp_path / "ok.wav")
    make_wav(tmp_path / "list.wav", extra_chunk=True)
    assert separation._wav_is_complete(tmp_path / "ok.wav") and separation._wav_is_complete(tmp_path / "list.wav")


def test_wav_is_complete_rejects_broken_files(tmp_path):
    make_wav(tmp_path / "ok.wav")
    raw = (tmp_path / "ok.wav").read_bytes()
    (tmp_path / "trunc.wav").write_bytes(raw[:4000])                                    # 檔頭說的長度比實際多
    zero = bytearray(raw); zero[40:44] = b"\0\0\0\0"; (tmp_path / "zero.wav").write_bytes(bytes(zero))       # 佔位的 0
    ff = bytearray(raw); ff[40:44] = b"\xff\xff\xff\xff"; (tmp_path / "ff.wav").write_bytes(bytes(ff))
    (tmp_path / "junk.wav").write_bytes(b"not a wav at all")
    for name in ("trunc", "zero", "ff", "junk", "nope"):
        assert not separation._wav_is_complete(tmp_path / f"{name}.wav"), name


# ---------------- 讀子程序輸出 ----------------
def test_iter_console_lines_splits_on_carriage_returns_and_keeps_multibyte_chars():
    data = "開始\r  0%|  | 0/10\r 50%|█████ | 5/10\r100%|██████████| 10/10\n完成 進度\n最後沒換行".encode("utf-8")
    assert list(separation._iter_console_lines(Drip(data))) == [
        "開始", "  0%|  | 0/10", " 50%|█████ | 5/10", "100%|██████████| 10/10", "完成 進度", "最後沒換行"]


# ---------------- 正常分離 ----------------
def test_normal_run(env):
    result, logs, prog = env.run()
    key = hashlib.sha1(f"{env.src.stat().st_size}:{env.src.stat().st_mtime_ns}".encode()).hexdigest()[:8]
    final = env.work / "htdemucs" / f"{env.src.stem}__{key}"
    assert Path(result) == final / "vocals.wav" and (final / "no_vocals.wav").exists()
    assert not list(env.work.glob("_partial_*"))                       # 暫存資料夾清掉了
    assert prog == sorted(prog) and prog[0] == 0.0 and prog[-1] == 1.0 and 0.6 in prog
    assert any("正在分離 測試影片 第1話.mp4" in x for x in logs) and not any("%|" in x for x in logs)   # 進度條不當成日誌
    argv = env.argv()
    assert argv[argv.index("--segment") + 1] == "7" and argv[argv.index("-d") + 1] == "cuda"
    assert Path(argv[argv.index("-o") + 1]).name.startswith("_partial_")


def test_second_run_reuses_the_result(env):
    first, _, _ = env.run()
    calls = env.count()
    second, logs, prog = env.run()
    assert second == first and env.count() == calls and prog == [1.0] and "跳過" in logs[0]


def test_same_name_different_content_is_separated_separately(env):
    first, _, _ = env.run()
    other = env.tmp / "other" / env.src.name
    other.parent.mkdir()
    other.write_bytes(b"y" * 2222)
    second, _, _ = env.run(other)
    assert Path(first) != Path(second) and Path(first).exists() and Path(second).exists() and env.count() == 2


def test_modified_time_change_triggers_a_new_separation(env):
    first, _, _ = env.run()
    st = env.src.stat()
    os.utime(env.src, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    second, _, _ = env.run()
    assert Path(first) != Path(second) and env.count() == 2


# ---------------- 舊版快取（<檔名>/vocals.wav）----------------
def legacy_case(env, name):
    src = env.tmp / f"{name}.mp4"
    src.write_bytes(b"z" * 500)
    legacy = env.work / "htdemucs" / src.stem
    legacy.mkdir(parents=True)
    make_wav(legacy / "vocals.wav")
    return src, legacy / "vocals.wav"


def test_legacy_cache_is_reused_when_complete_and_newer(env):
    src, wav = legacy_case(env, "舊版檔")
    os.utime(src, (time.time() - 100, time.time() - 100))
    result, _, _ = env.run(src)
    assert Path(result) == wav and env.count() == 0


def test_legacy_cache_is_ignored_when_truncated(env):
    src, wav = legacy_case(env, "殘檔")
    os.utime(src, (time.time() - 100, time.time() - 100))
    wav.write_bytes(wav.read_bytes()[:4000])
    result, _, _ = env.run(src)
    assert Path(result) != wav and env.count() == 1


def test_legacy_cache_is_ignored_when_source_is_newer(env):
    src, wav = legacy_case(env, "被換過")
    os.utime(wav, (time.time() - 1000, time.time() - 1000))
    result, _, _ = env.run(src)
    assert Path(result) != wav and env.count() == 1


# ---------------- 失敗 ----------------
@pytest.mark.parametrize("mode, exc, expect", [("fail", RuntimeError, "CUDA out of memory"), ("trunc", FileNotFoundError, "完整"),
                                                ("zero", FileNotFoundError, "完整")])
def test_failures_leave_no_result_or_partial_folder(env, monkeypatch, mode, exc, expect):
    monkeypatch.setenv("FAKE_MODE", mode)
    src = env.tmp / f"fail_{mode}.mp4"
    src.write_bytes(b"q" * 100)
    with pytest.raises(exc, match=expect):
        env.run(src)
    assert not list((env.work / "htdemucs").glob(f"{src.stem}__*")) and not list(env.work.glob("_partial_*"))


def test_incomplete_result_folder_from_before_is_replaced(env):
    src = env.tmp / "residue.mp4"
    src.write_bytes(b"r" * 321)
    key = hashlib.sha1(f"{src.stat().st_size}:{src.stat().st_mtime_ns}".encode()).hexdigest()[:8]
    residue = env.work / "htdemucs" / f"residue__{key}"
    residue.mkdir(parents=True)
    make_wav(residue / "vocals.wav")
    (residue / "vocals.wav").write_bytes((residue / "vocals.wav").read_bytes()[:3000])
    (residue / "leftover.txt").write_text("x")
    result, _, _ = env.run(src)
    assert separation._wav_is_complete(Path(result)) and not (residue / "leftover.txt").exists()


# ---------------- 裝置 ----------------
def test_auto_device_uses_cpu_without_gpu_and_says_so(env, monkeypatch):
    monkeypatch.setattr(separation, "resolve_device", lambda d: "cpu" if d == "auto" else d)
    _, logs, _ = env.run(device="auto")
    assert env.argv()[env.argv().index("-d") + 1] == "cpu" and any("改用 CPU" in x for x in logs)


def test_auto_device_uses_cuda_with_gpu(env, monkeypatch):
    monkeypatch.setattr(separation, "resolve_device", lambda d: "cuda" if d == "auto" else d)
    _, logs, _ = env.run(device="auto")
    assert env.argv()[env.argv().index("-d") + 1] == "cuda" and not any("改用 CPU" in x for x in logs)


def test_explicit_device_is_respected_even_without_gpu(env):
    _, logs, _ = env.run(device="cuda")
    assert env.argv()[env.argv().index("-d") + 1] == "cuda" and not any("改用 CPU" in x for x in logs)


def test_cpu_only_pytorch_gives_an_actionable_message(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "nocuda")
    with pytest.raises(RuntimeError) as e:
        env.run()
    assert "SEPARATION_DEVICE" in str(e.value) and "not compiled with CUDA" in str(e.value)


# ---------------- 取消 ----------------
def test_cancel_during_separation_stops_demucs_and_cleans_up(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "slow")
    monkeypatch.setenv("FAKE_PID", str(env.tmp / "pid.txt"))
    box, prog = {}, []

    def run():
        try:
            separation.separate_vocals(str(env.src), str(env.work), lambda m: None, device="cuda", progress_func=prog.append)
        except BaseException as e:
            box["exc"] = e
    thread = threading.Thread(target=run)
    thread.start()
    assert wait_for(lambda: prog and (env.tmp / "pid.txt").exists(), 15)
    pid = int((env.tmp / "pid.txt").read_text())
    assert alive(pid)
    t0 = time.time()
    JOBS.cancel()
    thread.join(10)
    assert isinstance(box.get("exc"), JobCancelled) and wait_dead(pid) and time.time() - t0 < 3
    assert not list((env.work / "htdemucs").glob("*")) if (env.work / "htdemucs").exists() else not list(env.work.glob("_partial_*"))


# ---------------- 找 Demucs 的順序 ----------------
def test_prefers_module_from_the_same_python(fake_demucs):
    cmd = separation._demucs_command()
    assert cmd[1:] == ["-m", "demucs.separate"]


def test_error_message_when_demucs_is_nowhere(monkeypatch):
    monkeypatch.setattr(separation.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(separation.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="pip install demucs"):
        separation._demucs_command()


def test_falls_back_to_demucs_on_path(monkeypatch, tmp_path):
    monkeypatch.setattr(separation.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(separation.shutil, "which", lambda name: str(tmp_path / "demucs.exe"))
    assert separation._demucs_command() == [str(tmp_path / "demucs.exe")]


# ---------------- 不跳黑視窗 ----------------
def test_process_is_started_without_a_console_window(env, monkeypatch):
    seen = {}
    real_spawn = JOBS.spawn

    def spy(cmd, **kw):
        seen.update(kw)
        kw.pop("creationflags", None)                      # 非 Windows 不能帶這個旗標
        return real_spawn(cmd, **kw)
    monkeypatch.setattr(separation, "_NO_WINDOW", 0x08000000)
    monkeypatch.setattr(JOBS, "spawn", spy)
    env.run()
    assert seen["creationflags"] == 0x08000000 and seen["stdin"] == separation.subprocess.DEVNULL
    assert seen["stdout"] == separation.subprocess.PIPE and seen["stderr"] == separation.subprocess.STDOUT
    assert seen["env"]["PYTHONIOENCODING"] == "utf-8"
