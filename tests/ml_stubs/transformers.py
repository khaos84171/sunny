"""測試用的假 transformers：模型輸出是「由音訊內容決定的固定亂數」，所以同樣輸入永遠得到同樣結果。
FAKE_SPECIAL_HOT=1 → 特殊符號（<s> </s> <unk> |）的機率被拉得異常高；FAKE_MODEL_DELAY=秒 → 每次推論的延遲。
FAKE_SCRIPT=劇本.json → 不再輸出亂數，改成照劇本在指定的時間放 CTC 的「尖峰」（像真的 CTC，字只出現在一兩個 frame）：
    {"audio": 音訊.npy 的路徑, "spikes": [[絕對時間秒, "字"], ...]}；用音訊內容找出這一段是整份音訊的哪裡。"""
import hashlib
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

CHARS = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだでどばびぶべぼぱぴぷぺぽっゃゅょー今日天気本地震熊"
VOCAB = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}
for _c in CHARS:
    VOCAB[_c] = len(VOCAB)


class _Tok:
    pad_token_id = 0
    all_special_ids = [0, 1, 2, 3]

    def get_vocab(self):
        return dict(VOCAB)


class Wav2Vec2Processor:
    tokenizer = _Tok()

    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def __call__(self, chunk, sampling_rate, return_tensors):
        return SimpleNamespace(input_values=torch._T(np.asarray(chunk)[None, :]))


_script_cache = {}


def _scripted_logits(chunk):
    path = os.environ["FAKE_SCRIPT"]
    if path not in _script_cache:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        _script_cache[path] = (np.load(data["audio"], mmap_mode="r"), data["spikes"])
    audio, spikes = _script_cache[path]
    offset = next(int(i) for i in np.flatnonzero(audio == chunk[0]) if np.array_equal(audio[i:i + 64], chunk[:64]))
    frames = (len(chunk) - 400) // 320 + 1          # 真的 wav2vec2 的 frame 數
    logits = np.full((frames, len(VOCAB)), -8.0)
    logits[:, 0] = 4.0
    for t, ch in spikes:
        f = int(round((t - offset / 16000) / 0.02))
        if 0 <= f < frames:
            logits[f, :] = -8.0
            logits[f, VOCAB[ch]] = 8.0
    return logits


class Wav2Vec2ForCTC:
    config = SimpleNamespace(conv_stride=(5, 2, 2, 2, 2, 2, 2))  # 跟真的 wav2vec2 一樣：每個 frame = 320 個取樣點

    @classmethod
    def from_pretrained(cls, name):
        torch._log("load " + name)
        return cls()

    def to(self, device):
        torch._log("to " + str(device))
        return self

    def eval(self):
        return self

    def __call__(self, x):
        time.sleep(float(os.environ.get("FAKE_MODEL_DELAY", "0")))
        chunk = x.a[0]
        if os.environ.get("FAKE_SCRIPT"):
            return SimpleNamespace(logits=torch._T(_scripted_logits(chunk)[None]))
        frames = max(len(chunk) // 320, 1)
        seed = int.from_bytes(hashlib.sha1(chunk[:2000].tobytes()).digest()[:4], "little")
        logits = np.random.default_rng(seed).normal(0, 1.6, size=(frames, len(VOCAB)))
        logits[:, 0] += 1.5  # blank 偏高，像真的 CTC 輸出
        if os.environ.get("FAKE_SPECIAL_HOT") == "1":
            logits[:, [1, 2, 3, 4]] += 9
        else:
            logits[:, [1, 2, 3, 4]] -= 8
        return SimpleNamespace(logits=torch._T(logits[None]))
