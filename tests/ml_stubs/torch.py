"""測試用的假 torch（只在測試啟動的 align_worker 子程序裡出現，用 PYTHONPATH 帶進去）。
用 numpy 實作 align_worker 會用到的那幾個功能。環境變數 FAKE_CUDA=1 → 假裝有 GPU；FAKE_MODEL_LOG=檔案 → 記錄模型搬移。"""
import os

import numpy as np


class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def to(self, *_):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __getitem__(self, i):
        return _T(self.a[i])


def _log(msg):
    if os.environ.get("FAKE_MODEL_LOG"):
        with open(os.environ["FAKE_MODEL_LOG"], "a", encoding="utf-8") as f:
            f.write(msg + "\n")


class cuda:
    @staticmethod
    def is_available():
        return os.environ.get("FAKE_CUDA") == "1"

    @staticmethod
    def empty_cache():
        _log("empty_cache")


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def inference_mode():
    return _Ctx()


def log_softmax(t, dim=-1):
    a = t.a - t.a.max(axis=dim, keepdims=True)
    return _T(a - np.log(np.exp(a).sum(axis=dim, keepdims=True)))
