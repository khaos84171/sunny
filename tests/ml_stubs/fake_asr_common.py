"""假 qwen_asr / nemo 共用：照劇本回答每一段聲音「聽到了什麼」。

測試音訊每一條字幕的位置放一段固定振幅的聲音（振幅 = 編號 / 100），其他地方是靜音；
假模型用片段裡的最大振幅認出這是第幾條，再查劇本 FAKE_ASR_SCRIPT（JSON 檔）：
    {"qwen": {"編號": "文字", ...}, "parakeet": {...}}；劇本沒寫的回傳空字串（＝聽不到）。
FAKE_MODEL_LOG=檔案 → 記錄載入、搬移、收到的提示；FAKE_ASR_MISSING=qwen,parakeet → 假裝這些套件沒裝。"""
import json
import os

import numpy as np


def log(msg):
    if os.environ.get("FAKE_MODEL_LOG"):
        with open(os.environ["FAKE_MODEL_LOG"], "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def check_installed(kind, module):
    if kind in os.environ.get("FAKE_ASR_MISSING", "").split(","):
        raise ImportError(f"No module named '{module}'", name=module)


def answer(kind, clip):
    idx = int(round(float(np.max(np.abs(np.asarray(clip)))) * 100)) if len(clip) else 0
    path = os.environ.get("FAKE_ASR_SCRIPT")
    script = json.load(open(path, encoding="utf-8")) if path else {}
    return script.get(kind, {}).get(str(idx), "")


class Module:
    def __init__(self, kind):
        self.kind = kind

    def to(self, where):
        log(f"{self.kind} to {where}")
        return self
