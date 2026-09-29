"""SRT 字幕檔的讀寫，以及開頭空白字幕。"""

from __future__ import annotations

import re
from pathlib import Path

from . import config


def format_timestamp(seconds: float) -> str:
    """將秒數轉換為 SRT 標準時間格式 HH:MM:SS,mmm"""
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, milliseconds = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


_SRT_TIME_RE = re.compile(
    r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
)


def _srt_time_to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def _read_text_any_encoding(path: Path) -> str:
    """字幕檔可能是 UTF-8（含 BOM）、UTF-16、Shift-JIS 或 Big5，依序嘗試。"""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "cp932", "cp950"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_srt(path) -> list[dict]:
    """讀取 SRT，回傳 [{"start", "end", "text"}, ...]；多行字幕保留換行。"""
    text = _read_text_any_encoding(Path(path)).replace("\r\n", "\n").replace("\r", "\n")
    segments = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.split("\n")
        for i, line in enumerate(lines):
            m = _SRT_TIME_RE.search(line)
            if m:
                g = m.groups()
                segments.append({
                    "start": _srt_time_to_sec(*g[:4]),
                    "end": _srt_time_to_sec(*g[4:]),
                    "text": "\n".join(l.strip() for l in lines[i + 1:] if l.strip()),
                })
                break
    return segments


_INVISIBLE_RE = re.compile(r"[\s\u200b\u200c\u200d\u2060\ufeff]")


def is_blank_text(text: str) -> bool:
    """只有空白或零寬字元（看不到任何字）的字幕文字。"""
    return not _INVISIBLE_RE.sub("", text)


def add_leading_blank(subs: list[dict], log_func) -> None:
    """
    在最前面插入一條從 00:00:00,000 開始、到第一句字幕出現為止的空白字幕（原地修改）。
    結束點剛好是第一句的起點，不會蓋到任何字幕。
    """
    if not subs:
        return
    first_start = min(sub["start"] for sub in subs)
    if first_start < config.LEADING_BLANK_MIN_SEC:
        log_func("第一句字幕從影片開頭就開始了，不需要插入空白字幕")
        return
    subs.insert(0, {"start": 0.0, "end": first_start, "text": config.LEADING_BLANK_TEXT})
    log_func(f"已在最前面插入空白字幕（00:00:00,000 --> {format_timestamp(first_start)}）")


def write_srt(path, segments: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            f.write(
                f"{idx}\n{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n"
                f"{seg['text']}\n\n"
            )
