"""記住視窗上的勾選狀態：讀寫設定檔。"""
import json

from whisper_app import config, settings_store


def test_roundtrip_uses_config_path_by_default():
    data = {"version": 1, "use_sep": False, "hotwords": {"クイズ": True, "伊沢拓司": False}}
    assert settings_store.save_settings(data) is True
    assert config.SETTINGS_FILE.exists()
    assert settings_store.load_settings() == data
    assert not config.SETTINGS_FILE.with_name(config.SETTINGS_FILE.name + ".tmp").exists()   # 暫存檔已換名


def test_missing_file_gives_empty_settings():
    assert settings_store.load_settings() == {}


def test_corrupt_json_gives_empty_settings(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{這不是json", encoding="utf-8")
    assert settings_store.load_settings(path) == {}


def test_non_object_json_gives_empty_settings(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert settings_store.load_settings(path) == {}


def test_save_to_unwritable_place_returns_false(tmp_path):
    assert settings_store.save_settings({"a": 1}, tmp_path / "沒有這個資料夾" / "s.json") is False


def test_save_keeps_previous_file_when_it_fails(tmp_path, monkeypatch):
    path = tmp_path / "s.json"
    settings_store.save_settings({"v": 1}, path)

    def boom(*args, **kwargs):
        raise PermissionError("被占用")

    monkeypatch.setattr(settings_store.os, "replace", boom)
    assert settings_store.save_settings({"v": 2}, path) is False
    assert json.loads(path.read_text(encoding="utf-8")) == {"v": 1}      # 原本的沒被弄壞
