"""測試共用的小工具。"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent


def alive(pid: int) -> bool:
    """這個 pid 的程序還活著嗎（Linux：殭屍程序算已經結束）。"""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state not in ("Z", "X")
    except OSError:
        return False


def wait_dead(pid: int, timeout: float = 5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return False


def wait_for(cond, timeout: float = 8.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.03)
    return False


def subprocess_env(**extra) -> dict:
    """給測試啟動的子程序用：讓它們找得到專案與假的模組。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=os.pathsep.join(
        [str(TESTS / "fakes"), str(TESTS / "ml_stubs"), str(ROOT)]))
    env.update({k: str(v) for k, v in extra.items()})
    return env


def make_align_job(out_dir: Path, seed: int = 7, device: str = "cuda") -> Path:
    """產生一個確定性的對齊工作（40 秒雜訊音訊 + 12 條字幕），回傳 job.json 的路徑。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    np.save(out_dir / "audio.npy", rng.normal(0, 0.1, size=16000 * 40).astype(np.float32))
    texts = ["今日は天気がいい", "熊本地震", "あいうえおかきくけこ", "ABC 123 です", "さしすせそたちつてとなにぬねのはひふへほ",
             "まみむめもやゆよらりるれろわをん", "ああ", "がぎぐげござじずぜぞだでどばびぶべぼぱぴぷぺぽ", "。。。",
             "きょうはてんきがいいですねー",
             "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞ",
             "本日は晴天なり"]
    import random
    random.Random(seed).shuffle(texts)
    segments, t = [], 1.0 + (seed % 5) * 0.37
    for i, text in enumerate(texts):
        d = 0.4 + 0.15 * len(text)
        seg = {"start": round(t, 2), "end": round(t + d, 2), "text": text}
        if i % 3 == 0 and len(text) > 2:
            seg["words"] = [text[:2], text[2:]]
        segments.append(seg)
        t += d + 0.5
    job = {"audio_npy": str(out_dir / "audio.npy"), "segments": segments, "model_name": "fake/wav2vec2", "device": device,
           "sample_rate": 16000,
           "params": {"pad_sec": 0.4, "edge_sec": 0.12, "wide_sec": 3.0, "batch_sec": 20.0, "low_conf": 0.3, "max_inner_gap": 1.0}}
    path = out_dir / "job.json"
    path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    return path


def run_worker(args, stdin_lines=(), env_extra=None, close_stdin=True, timeout=90):
    """啟動 align_worker.py，寫入 stdin 的每一行，回傳 (結束碼, 全部輸出, 花費秒數)。"""
    import subprocess
    proc = subprocess.Popen([sys.executable, str(ROOT / "align_worker.py"), *map(str, args)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            env=subprocess_env(**(env_extra or {})))
    for line in stdin_lines:
        proc.stdin.write(line if line.endswith("\n") else line + "\n")
    proc.stdin.flush()
    if close_stdin:
        proc.stdin.close()
    t0 = time.time()
    out = proc.stdout.read()
    proc.wait(timeout=timeout)
    return proc.returncode, out, time.time() - t0
