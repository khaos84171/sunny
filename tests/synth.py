"""合成「像人聲」的測試音訊：背景雜訊 + 每個詞一段有基頻諧波的母音（前面可以帶一小段較弱的子音雜訊）。
詞的真正起訖是我們決定的，所以能拿來驗證「聲音能量微調」有沒有把邊界貼回真正的位置。"""
import numpy as np

SR = 16000


def synth_speech(total_sec, words, floor_db=-60.0, bgm_db=None, seed=0, f0=120.0):
    """
    words = [(起點秒, 終點秒, 子音秒數), ...]：聲音從「起點」開始（先是子音雜訊，再接母音），到「終點」結束。
    floor_db = 背景白雜訊的音量；bgm_db = 不為 None 時再疊一條持續的 220 Hz 弦波（當背景音樂）。
    """
    rng = np.random.default_rng(seed)
    n = int(total_sec * SR)
    t = np.arange(n) / SR
    x = rng.normal(0, 10 ** (floor_db / 20), n)
    if bgm_db is not None:
        x += 10 ** (bgm_db / 20) * np.sin(2 * np.pi * 220 * t)
    for onset, offset, consonant in words:
        i0, i1 = int(round(onset * SR)), int(round(offset * SR))
        tt = t[i0:i1]
        vowel = sum(np.sin(2 * np.pi * h * f0 * tt) / h for h in range(1, 34)) * 0.15
        shape = np.ones(len(tt))
        attack, release = int(0.015 * SR), int(0.04 * SR)
        shape[:attack] = np.linspace(0, 1, attack)
        shape[-release:] = np.linspace(1, 0, release)
        c = int(round(consonant * SR))
        sig = np.zeros(len(tt))
        sig[c:] = (vowel * shape)[c:]
        if c:
            sig[:c] = rng.normal(0, 0.015, c) * np.linspace(0.5, 1, c)
        x[i0:i1] += sig
    return x.astype(np.float32)
