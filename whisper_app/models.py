"""各模組共用的資料型別。"""

from __future__ import annotations

import collections


# 一個詞的時間與文字。Whisper 給的詞、以及對齊後改用 wav2vec2 時間的詞都用這個格式
Word = collections.namedtuple("Word", "start end word")
