"""整理拖進來的東西:展開資料夾、過濾非影片／音訊、去掉重複。"""

from __future__ import annotations

import os
import re
from pathlib import Path

from . import config


def _natural_key(path) -> list:
    """檔名裡的數字照大小排（第2話 排在 第10話 前面）。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path))]


def collect_dropped_files(paths: list[str]) -> tuple[list[str], list[str], list[str]]:
    """
    整理拖進來的東西，回傳 (影片／音訊, .srt, 略過的說明)。
    資料夾會展開成裡面（含子資料夾）的影片／音訊，依檔名排序；資料夾裡的 .srt 不算，
    避免拖資料夾就意外進了「只對齊」模式。不存在的、不是影片／音訊的、以「.」開頭的隱藏檔
    （例如 ._xxx.mp4）都略過；同一個檔案不管拖幾次只算一次。
    """
    media, srts, skipped, seen = [], [], [], set()

    def add(bucket: list, p: Path) -> None:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            bucket.append(str(p))

    for raw in paths:
        p = Path(raw)
        suffix = p.suffix.lower()
        if p.is_dir():
            found = sorted((f for f in p.rglob("*")
                            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in config.MEDIA_EXTENSIONS),
                           key=_natural_key)
            if not found:
                skipped.append(f"{p.name}（資料夾裡沒有影片或音訊）")
            for f in found:
                add(media, f)
        elif not p.is_file():
            skipped.append(f"{p.name or raw}（找不到這個檔案）")
        elif p.name.startswith("."):
            skipped.append(p.name)
        elif suffix == ".srt":
            add(srts, p)
        elif suffix in config.MEDIA_EXTENSIONS:
            add(media, p)
        else:
            skipped.append(p.name)
    return media, srts, skipped
