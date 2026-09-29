"""整理拖進來的東西：資料夾展開、過濾、去重複、排序。"""
from pathlib import Path

from whisper_app.dropped import collect_dropped_files


def touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"1")
    return str(path)


def names(paths):
    return [Path(p).name for p in paths]


def test_folder_is_expanded_recursively_with_natural_sort(tmp_path):
    root = tmp_path / "S1"
    for name in ["第2話.mp4", "第10話.mp4", "第1話.MKV", "note.txt", "cover.jpg", "._第1話.mp4", "字幕.srt"]:
        touch(root / name)
    touch(root / "sub" / "extra.m4a")
    media, srts, skipped = collect_dropped_files([str(root)])
    assert [n for n in names(media) if n.startswith("第")] == ["第1話.MKV", "第2話.mp4", "第10話.mp4"]   # 數字照大小排
    assert set(names(media)) == {"第1話.MKV", "第2話.mp4", "第10話.mp4", "extra.m4a"}
    assert srts == [] and skipped == []                     # 資料夾裡的 .srt、隱藏檔、txt、jpg 靜靜略過


def test_explicit_files_are_classified_and_reported(tmp_path):
    video, srt, txt = touch(tmp_path / "a.mp4"), touch(tmp_path / "a.srt"), touch(tmp_path / "readme.txt")
    media, srts, skipped = collect_dropped_files([video, srt, txt, str(tmp_path / "不存在.mp4")])
    assert media == [video] and srts == [srt]
    assert any("readme.txt" in s for s in skipped) and any("找不到" in s for s in skipped)


def test_duplicates_count_once(tmp_path):
    video = touch(tmp_path / "a.wav")
    media, _, _ = collect_dropped_files([video, video, str(tmp_path / "." / "a.wav")])
    assert media == [video]


def test_empty_folder_is_reported(tmp_path):
    touch(tmp_path / "empty" / "x.txt")
    media, srts, skipped = collect_dropped_files([str(tmp_path / "empty")])
    assert media == [] and any("資料夾裡沒有" in s for s in skipped)


def test_empty_input():
    assert collect_dropped_files([]) == ([], [], [])


def test_hidden_file_is_skipped(tmp_path):
    hidden = touch(tmp_path / ".hidden.mp4")
    assert collect_dropped_files([hidden]) == ([], [], [".hidden.mp4"])
