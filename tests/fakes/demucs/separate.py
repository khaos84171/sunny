"""假的 Demucs（python -m demucs.separate）：輸出 tqdm 風格的進度（用 \\r 原地更新），寫出真的 WAV。
環境變數 FAKE_MODE：ok / fail（結束碼 1）/ trunc（WAV 被截斷）/ zero（WAV 檔頭長度是 0）/ slow（卡在 40%）/ nocuda。
FAKE_LOG=檔案 → 記下收到的參數；FAKE_PID=檔案 → 記下自己的 pid。"""
import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path


def write_wav(path, seconds=1.0, rate=8000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(rate * seconds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--two-stems")
    ap.add_argument("-n")
    ap.add_argument("-d")
    ap.add_argument("-o")
    ap.add_argument("--segment")
    ap.add_argument("track")
    a = ap.parse_args()
    if os.environ.get("FAKE_LOG"):
        with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as f:
            f.write(json.dumps(sys.argv[1:], ensure_ascii=False) + "\n")
    mode = os.environ.get("FAKE_MODE", "ok")
    err = sys.stderr
    print("Selected model is a bag of 1 models. You will see that many progress bars per track.", file=err, flush=True)
    print(f"正在分離 {Path(a.track).name}", file=err, flush=True)
    if os.environ.get("FAKE_PID"):
        Path(os.environ["FAKE_PID"]).write_text(str(os.getpid()))
    for i in range(0, 101, 20):
        if mode == "slow" and i >= 40:
            time.sleep(60)
        err.write(f"\r{i:3d}%|{'█' * (i // 10)}{' ' * (10 - i // 10)}| {i}.0/100.0 [00:0{i // 20}<00:00,  1.0seconds/s]")
        err.flush()
        time.sleep(0.03)
    err.write("\n")
    err.flush()
    if mode == "nocuda":
        print("AssertionError: Torch not compiled with CUDA enabled", file=err, flush=True)
        sys.exit(1)
    if mode == "fail":
        print("RuntimeError: CUDA out of memory (模擬)", file=err, flush=True)
        sys.exit(1)
    out = Path(a.o) / a.n / Path(a.track).name.rsplit(".", 1)[0]
    out.mkdir(parents=True)
    write_wav(out / "vocals.wav")
    write_wav(out / "no_vocals.wav")
    if mode == "trunc":  # 檔頭說有 1 秒資料，實際只剩一半
        p = out / "vocals.wav"
        p.write_bytes(p.read_bytes()[:4000])
    if mode == "zero":  # 中斷時常見：檔頭長度還是佔位的 0
        p = out / "vocals.wav"
        b = bytearray(p.read_bytes())
        b[40:44] = b"\x00\x00\x00\x00"
        p.write_bytes(bytes(b))


if __name__ == "__main__":
    main()
