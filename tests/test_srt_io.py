"""SRT 讀寫：時間格式、各種編碼、多行字幕、空白字幕。"""
import pytest

from whisper_app import srt_io


@pytest.mark.parametrize("seconds, expected", [
    (0, "00:00:00,000"), (3661.5, "01:01:01,500"), (59.9996, "00:01:00,000"),
    (-1, "00:00:00,000"),                      # 負數當作 0
    (36000.25, "10:00:00,250"),
])
def test_format_timestamp(seconds, expected):
    assert srt_io.format_timestamp(seconds) == expected


def test_parse_srt_basic_multiline_and_blank_cue(tmp_path):
    path = tmp_path / "a.srt"
    path.write_bytes("1\r\n00:00:01,000 --> 00:00:02,500\r\nこんにちは\r\n世界\r\n\r\n"
                     "2\r\n00:00:03,000 --> 00:00:04,000\r\n​\r\n\r\n"
                     "3\r\n00:01:00.5 --> 00:01:02.25\r\nおわり\r\n".encode("utf-8-sig"))
    assert srt_io.parse_srt(path) == [
        {"start": 1.0, "end": 2.5, "text": "こんにちは\n世界"},   # 多行字幕保留換行
        {"start": 3.0, "end": 4.0, "text": "​"},
        {"start": 60.5, "end": 62.25, "text": "おわり"},           # 也認得用「.」當小數點的時間
    ]


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "cp932"])
def test_parse_srt_encodings(tmp_path, encoding):
    path = tmp_path / "enc.srt"
    path.write_bytes("1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n".encode(encoding))
    assert srt_io.parse_srt(path) == [{"start": 1.0, "end": 2.0, "text": "こんにちは"}]


def test_parse_srt_ignores_garbage_blocks(tmp_path):
    path = tmp_path / "g.srt"
    path.write_text("這不是字幕\n\n1\n00:00:01,000 --> 00:00:02,000\nok\n", encoding="utf-8")
    assert [s["text"] for s in srt_io.parse_srt(path)] == ["ok"]


def test_write_then_parse_roundtrip(tmp_path):
    subs = [{"start": 0, "end": 1.5, "text": "x\ny"}, {"start": 2, "end": 3.25, "text": "z"}]
    path = tmp_path / "o.srt"
    srt_io.write_srt(path, subs)
    assert path.read_bytes() == b"1\n00:00:00,000 --> 00:00:01,500\nx\ny\n\n2\n00:00:02,000 --> 00:00:03,250\nz\n\n"
    assert srt_io.parse_srt(path) == subs


@pytest.mark.parametrize("text, blank", [("​ \n", True), ("", True), ("﻿⁠", True), ("あ", False), (" a ", False)])
def test_is_blank_text(text, blank):
    assert srt_io.is_blank_text(text) is blank


def test_add_leading_blank_inserts_until_first_subtitle():
    subs, logs = [{"start": 2.0, "end": 3.0, "text": "a"}], []
    srt_io.add_leading_blank(subs, logs.append)
    assert subs[0] == {"start": 0.0, "end": 2.0, "text": "​"} and subs[1]["text"] == "a"
    assert "00:00:02,000" in logs[0]


def test_add_leading_blank_starts_at_the_timeline_origin():
    """時間軸從 01:00:00 開始的字幕：空白字幕從那裡開始，不是從 0（不然是一條長一小時的空白字幕）。"""
    subs = [{"start": 3603.5, "end": 3605.0, "text": "a"}]
    logs = []
    srt_io.add_leading_blank(subs, logs.append, origin=3600.0)
    assert (subs[0]["start"], subs[0]["end"]) == (3600.0, 3603.5) and "01:00:00,000 --> 01:00:03,500" in logs[0]
    subs = [{"start": 3600.0, "end": 3601.0, "text": "a"}]                # 一開始就有字幕：不用插
    srt_io.add_leading_blank(subs, logs.append, origin=3600.0)
    assert len(subs) == 1


def test_add_leading_blank_skipped_when_already_at_start():
    subs, logs = [{"start": 0.01, "end": 3.0, "text": "a"}], []
    srt_io.add_leading_blank(subs, logs.append)
    assert len(subs) == 1 and "不需要" in logs[0]


def test_add_leading_blank_empty_list_is_noop():
    subs = []
    srt_io.add_leading_blank(subs, lambda m: None)
    assert subs == []
