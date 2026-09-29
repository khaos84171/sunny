"""Whisper 字幕產生器的程式碼。入口是上一層資料夾的 w1_1.py；可以調整的設定都在 config.py。"""

import os

# 處理 OpenMP 衝突（要在載入 faster_whisper 之前設定）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
