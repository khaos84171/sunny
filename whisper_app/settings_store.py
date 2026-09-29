"""記住視窗上的勾選狀態（whisper_settings.json）。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import config


def load_settings(path: Path | None = None) -> dict:
    """讀取上次記住的設定（勾選狀態、Hotwords）。檔案不存在、壞掉、格式不對都當作沒有設定，全部用預設值。"""
    try:
        data = json.loads((path or config.SETTINGS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(data: dict, path: Path | None = None) -> bool:
    """存設定。先寫暫存檔再換名，程式中途當掉也不會留下寫到一半的檔案。成功回傳 True。"""
    path = path or config.SETTINGS_FILE
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError:
        return False
