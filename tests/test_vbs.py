"""字幕產生器.vbs：靜態檢查（沒辦法在這裡執行它），以及它試跑 Python 用的那段程式碼的約定。"""
import re
import subprocess
import sys

import pytest

from helpers import ROOT
from vbs_lint import lint

VBS = ROOT / "字幕產生器.vbs"


@pytest.fixture(scope="module")
def text():
    return VBS.read_bytes().decode("ascii")


def test_the_launcher_passes_the_static_checks(text):
    assert lint(text) == []


# ---------------- 檢查器本身要抓得到常見的錯 ----------------
def broken(text, old, new):
    assert old in text
    return text.replace(old, new, 1)


@pytest.mark.parametrize("old, new, expected", [
    ("        chosenExe = parts(0)\r\n", "        chosenExe = parts(0)\r\n        chosenArgz = parts(1)\r\n", "chosenArgz"),      # 變數名稱打錯
    ("    End If\r\n    WScript.Quit 1\r\nEnd If\r\n\r\n' ----------------------------------------------------------------", "    WScript.Quit 1\r\nEnd If\r\n\r\n' ----------------------------------------------------------------", "if 沒有結尾"),   # 少一個 End If
    ("Next\r\nEnd Sub\r\n\r\n' Add the Python installs", "End Sub\r\n\r\n' Add the Python installs", "沒有結尾"),           # 少一個 Next
    ("APP_TITLE = \"Whisper Subtitle Generator\"", "APP_TITLE = \"Whisper 字幕產生器\"", "非 ASCII"),
    ("Option Explicit\r\n", "", "Option Explicit"),
    ("Dim objShell, objFSO,", "Dim objFSO,", "objShell"),                                                  # 少宣告一個
])
def test_the_checker_catches_typical_mistakes(text, old, new, expected):
    problems = lint(broken(text, old, new))
    assert problems and any(expected.lower() in p.lower() for p in problems), problems


def test_the_checker_catches_bare_line_feeds(text):
    assert any("LF" in p for p in lint(text.replace("\r\n", "\n")))


# ---------------- 它依賴的檔案和約定 ----------------
def test_the_files_it_checks_for_exist():
    for name in re.findall(r'BuildPath\(strScriptDir, "([^"]+)"\)', VBS.read_text(encoding="ascii")):
        assert (ROOT / name.replace("\\", "/")).exists(), name


def probe_code(text: str, major: int, minor: int) -> str:
    """照 .vbs 裡的樣子組出試跑 Python 用的那段程式碼。"""
    template = re.search(r'code = "(.*?)"\r\n', text).group(1)
    return template.replace('" & MIN_MAJOR & "', str(major)).replace('" & MIN_MINOR & "', str(minor))


def run_probe(code: str) -> int:
    return subprocess.run([sys.executable, "-c", code]).returncode


def test_probe_code_exit_codes_match_what_the_launcher_expects(text):
    major, minor = (int(re.search(rf"Const MIN_{n} = (\d+)", text).group(1)) for n in ("MAJOR", "MINOR"))
    assert (major, minor) == (3, 9)                                       # 跟 w1_1.py 說明的最低版本一致
    assert run_probe(probe_code(text, major, minor)) == 0                 # 夠新 → 0（VBS 當作「可以用」）
    assert run_probe(probe_code(text, 99, 0)) == 3                        # 太舊 → 3（VBS 當作「版本太舊」）
    assert run_probe("import sys; sys.exit(1)") not in (0, 3)             # 其他結束碼 → VBS 當作「不能用」
