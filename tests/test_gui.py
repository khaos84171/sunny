"""介面：設定記憶、Hotwords 預設、日誌視窗、取消／佇列、拖放、錯誤回報、關閉視窗、DPI。
需要真的 tkinter 和顯示器；沒有的話整個檔案會跳過（Linux 可以用 xvfb-run -a python -m pytest）。"""
import json
import time
import types
from pathlib import Path

import pytest

tkinter = pytest.importorskip("tkinter")

from helpers import wait_for                                                              # noqa: E402
from whisper_app import config, gui, runtime                                              # noqa: E402
from whisper_app.jobs import JOBS                                                         # noqa: E402

NAMES, COMMON = config.HOTWORD_NAME_CANDIDATES, config.HOTWORD_COMMON_CANDIDATES


@pytest.fixture
def make_app(monkeypatch, tmp_path):
    """建立真的 App。settings = 先寫進設定檔的 dict；raw = 先寫進設定檔的原始文字。測試結束時關掉所有視窗。"""
    roots = []
    dialogs = types.SimpleNamespace(calls=[], answers={"askokcancel": True, "askyesno": True})

    def fake(kind):
        def ask(title, message, **kw):
            dialogs.calls.append((kind, title, message))
            return dialogs.answers.get(kind)
        return ask
    for kind in ("askokcancel", "askyesno", "showerror"):
        monkeypatch.setattr(gui.messagebox, kind, fake(kind))
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **kw: "")

    def make(settings=None, raw=None):
        if settings is not None:
            config.SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False), encoding="utf-8")
        if raw is not None:
            config.SETTINGS_FILE.write_text(raw, encoding="utf-8")
        try:
            root = gui.TkinterDnD.Tk()
        except tkinter.TclError as e:
            pytest.skip(f"沒有可用的顯示器：{e}")
        roots.append(root)
        app = gui.App(root)
        spin(root, 0.15)
        return app
    make.dialogs = dialogs
    yield make
    JOBS.cancel()
    for root in roots:
        try:
            root.destroy()
        except tkinter.TclError:
            pass


def spin(root, seconds=0.15):
    end = time.time() + seconds
    while time.time() < end:
        root.update()
        time.sleep(0.01)


def queued_logs(app):
    """取出（並清掉）還在佇列裡、尚未寫進視窗的日誌。只有沒在跑事件迴圈時才準。"""
    out = []
    while not app.ui_queue.empty():
        kind, payload = app.ui_queue.get_nowait()
        if kind == "log":
            out.append(str(payload))
    return out


def window_text(app):
    return app.log_box.get("1.0", "end-1c")


# ================= 設定記憶與 Hotwords 預設 =================
def test_first_run_defaults(make_app):
    app = make_app()
    assert all(v.get() for v in (app.use_sep_var, app.use_align_var, app.use_split_var, app.add_blank_var))
    assert not any(app.hotword_vars[w].get() for w in NAMES)              # 人名每部影片不同：預設不勾
    assert all(app.hotword_vars[w].get() for w in COMMON)                 # 通用詞：預設勾
    assert not config.SETTINGS_FILE.exists()                              # 沒改任何東西就不寫檔


def test_saved_settings_are_restored(make_app):
    app = make_app({"version": 1, "use_sep": False, "use_align": True, "use_split": False, "add_blank": False,
                    "hotwords": {"伊沢拓司": True, "クイズ": False, "不存在的詞": True}})
    assert (app.use_sep_var.get(), app.use_align_var.get(), app.use_split_var.get(), app.add_blank_var.get()) == (False, True, False, False)
    assert app.hotword_vars["伊沢拓司"].get() is True and app.hotword_vars["クイズ"].get() is False
    assert app.hotword_vars["ふくら"].get() is False and app.hotword_vars["東大"].get() is True      # 沒記到的用預設
    assert "不存在的詞" not in app.hotword_vars


@pytest.mark.parametrize("kw", [dict(raw="{這不是json"), dict(raw="[1,2,3]"), dict(settings={"hotwords": ["a"], "use_sep": "yes"})])
def test_broken_settings_fall_back_to_defaults(make_app, kw):
    app = make_app(**kw)
    assert app.use_align_var.get() is True and app.hotword_vars["ふくら"].get() is False and app.hotword_vars["クイズ"].get() is True


def test_change_is_saved_after_a_short_delay(make_app):
    app = make_app()
    app.use_split_var.set(False)
    assert not config.SETTINGS_FILE.exists()                              # 不會立刻存
    spin(app.root, 0.8)
    data = json.loads(config.SETTINGS_FILE.read_text(encoding="utf-8"))
    assert data["version"] == 1 and data["use_split"] is False and data["use_sep"] is True
    assert data["hotwords"]["クイズ"] is True and data["hotwords"]["ふくら"] is False


def test_select_all_and_none_saves_only_once(make_app, monkeypatch):
    app = make_app()
    saved = []
    real = gui.save_settings
    monkeypatch.setattr(gui, "save_settings", lambda data, path=None: (saved.append(data), real(data, path))[1])
    for state in (True, False, True):
        app._set_all_hotwords(state)
    spin(app.root, 0.9)
    assert len(saved) == 1 and all(saved[0]["hotwords"].values())


def test_closing_the_window_saves_immediately_and_next_start_restores_it(make_app, monkeypatch):
    app = make_app()
    app.add_blank_var.set(False)
    app.root.destroy = lambda: None
    app._on_close()
    assert json.loads(config.SETTINGS_FILE.read_text(encoding="utf-8"))["add_blank"] is False
    again = make_app()
    assert again.add_blank_var.get() is False


def test_unwritable_settings_warn_once_without_crashing(make_app, tmp_path):
    app = make_app()
    config.SETTINGS_FILE = tmp_path / "沒有這個資料夾" / "s.json"
    for state in (False, True):
        app.use_sep_var.set(state)
        spin(app.root, 0.8)
    assert window_text(app).count("無法儲存設定") == 1


# ================= 日誌視窗 =================
def test_log_window_keeps_only_the_newest_lines(make_app):
    app = make_app()
    limit = config.LOG_MAX_LINES
    app._append_log([f"line {i}" for i in range(limit + 2000)])
    spin(app.root)
    assert int(app.log_box.index("end-1c").split(".")[0]) - 1 == limit
    assert app.log_box.get("1.0", "1.end") == "line 2000" and app.log_box.get(f"{limit}.0", f"{limit}.end") == f"line {limit + 1999}"
    app._append_log(["A", "B", "C"])
    spin(app.root)
    assert app.log_box.get("1.0", "1.end") == "line 2003" and app.log_box.get(f"{limit - 2}.0", f"{limit}.end") == "A\nB\nC"
    assert str(app.log_box.cget("state")) == "disabled"                   # 唯讀


def test_log_follows_new_messages_only_when_scrolled_to_the_bottom(make_app):
    app = make_app()
    box = app.log_box
    app._append_log([f"row {i}" for i in range(300)])
    spin(app.root)
    assert box.yview()[1] >= 0.999                                        # 在最底下：跟著捲到最新
    box.yview_moveto(0.2)
    spin(app.root)
    top = box.get("@0,0", "@0,0 lineend")
    app._append_log([f"later {i}" for i in range(20)])
    spin(app.root)
    assert box.get("@0,0", "@0,0 lineend") == top and box.yview()[1] < 0.99      # 往上捲去看舊訊息：不會被拉回底部
    box.yview_moveto(1.0)
    spin(app.root)
    app._append_log(["back"])
    spin(app.root)
    assert box.yview()[1] >= 0.999                                        # 捲回底下：又恢復跟著捲


def test_messages_flow_from_the_queue_in_order_with_progress_and_status(make_app):
    app = make_app()
    for i in range(50):
        app.log(f"via queue {i}")
    app.set_progress(0.5)
    app.set_status("跑跑")
    spin(app.root)
    lines = [l for l in window_text(app).splitlines() if l.startswith("via queue")]
    assert lines == [f"via queue {i}" for i in range(50)] and app.progress["value"] == 50.0 and "跑跑" in app.status_var.get()


def test_poll_keeps_rescheduling_after_an_error(make_app):
    app = make_app()
    scheduled = []
    app.root.after = lambda *a, **k: scheduled.append(a)
    app.ui_queue.put(("progress", "不是數字"))
    with pytest.raises(ValueError):
        app.poll_ui_queue()
    assert len(scheduled) == 1                                             # 出錯了計時器還在，畫面不會從此不更新
    app.ui_queue.put(("progress", 0.25))
    app.poll_ui_queue()
    assert app.progress["value"] == 25.0


def test_startup_notes_are_written_to_the_window(make_app):
    runtime.STARTUP_NOTES.append("注意：測試用的啟動訊息")
    app = make_app()
    assert "注意：測試用的啟動訊息" in window_text(app)


# ================= 取消 / 佇列 / 拖放 =================
@pytest.fixture
def fake_jobs(monkeypatch):
    calls = types.SimpleNamespace(started=[], finished=[])

    def process(path, *a, **k):
        name = Path(path).name
        calls.started.append(name)
        if name.startswith("slow"):
            for _ in range(400):
                JOBS.check()
                time.sleep(0.02)
        if name.startswith("boom"):
            raise ValueError("boom 模擬")
        calls.finished.append(name)

    def align_only(srt, media, *a, **k):
        calls.started.append(Path(srt).name)
        calls.finished.append(Path(srt).name)
    monkeypatch.setattr(gui, "process_file", process)
    monkeypatch.setattr(gui, "align_existing_srt", align_only)
    return calls


def touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"1")
    return str(path)


def test_dropping_a_folder_expands_it_and_reports_what_was_skipped(make_app, fake_jobs, tmp_path):
    app = make_app()
    media = tmp_path / "media"
    for name in ("a.mp4", "b.mp3", "sub/c.mkv", "readme.txt", "pic.jpg"):
        touch(media / name)
    app._handle_dropped([str(media), str(media / "a.mp4"), str(media / "pic.jpg"), str(tmp_path / "不存在.mp4")])
    assert wait_for(lambda: len(fake_jobs.finished) == 3) and sorted(fake_jobs.finished) == ["a.mp4", "b.mp3", "c.mkv"]
    text = "\n".join(queued_logs(app))
    assert "略過 2 個" in text and "pic.jpg" in text and "不存在.mp4" in text


def test_dropping_only_unsupported_files_queues_nothing(make_app, fake_jobs, tmp_path):
    app = make_app()
    app._handle_dropped([touch(tmp_path / "readme.txt")])
    assert "沒有可以處理" in "\n".join(queued_logs(app)) and app._pending == 0 and not fake_jobs.started


def test_many_files_ask_for_confirmation_first(make_app, fake_jobs, tmp_path):
    app = make_app()
    for i in range(config.MANY_FILES_CONFIRM + 1):
        touch(tmp_path / "many" / f"v{i:02d}.mp4")
    make_app.dialogs.answers["askyesno"] = False
    app._handle_dropped([str(tmp_path / "many")])
    assert any(c[0] == "askyesno" for c in make_app.dialogs.calls) and "已取消加入" in "\n".join(queued_logs(app))
    assert app._pending == 0 and not fake_jobs.started
    make_app.dialogs.answers["askyesno"] = True
    app._handle_dropped([str(tmp_path / "many")])
    assert wait_for(lambda: len(fake_jobs.finished) == config.MANY_FILES_CONFIRM + 1)


def test_srt_plus_video_runs_align_only(make_app, fake_jobs, tmp_path):
    app = make_app()
    srt, video = touch(tmp_path / "a_large-v3_sep.srt"), touch(tmp_path / "a.mp4")
    app._handle_dropped([srt, video])
    assert wait_for(lambda: fake_jobs.finished == ["a_large-v3_sep.srt"])


def test_cancel_stops_the_current_file_clears_the_queue_and_later_files_still_work(make_app, fake_jobs, tmp_path):
    app = make_app()
    app._handle_dropped([touch(tmp_path / f"slow{i}.mp4") for i in (1, 2, 3)])
    assert wait_for(lambda: fake_jobs.started == ["slow1.mp4"]) and app._pending == 3
    t0 = time.time()
    app.cancel_all()
    assert wait_for(lambda: app._pending == 0, 5) and time.time() - t0 < 2
    assert fake_jobs.started == ["slow1.mp4"] and fake_jobs.finished == []
    text = "\n".join(queued_logs(app))
    assert "清除排隊中的 2 個" in text and "已取消：slow1.mp4" in text
    fake_jobs.started.clear()
    app._handle_dropped([touch(tmp_path / "after.mp4")])
    assert wait_for(lambda: fake_jobs.finished == ["after.mp4"])          # 取消之後新排的工作正常執行


def test_cancel_with_nothing_running_only_says_so(make_app, fake_jobs):
    app = make_app()
    app.cancel_all()
    assert "目前沒有處理中" in "\n".join(queued_logs(app)) and app._pending == 0


def test_a_job_queued_before_a_cancel_is_skipped_by_the_worker(make_app, fake_jobs, tmp_path):
    app = make_app()
    stale = {"kind": "transcribe", "path": str(tmp_path / "stale.mp4"), "use_sep": False, "use_align": False, "use_split": False,
             "hotwords": [], "add_blank": False, "gen": JOBS.generation - 1}
    app._pending += 1
    app.file_queue.put(stale)
    assert wait_for(lambda: app._pending == 0) and not fake_jobs.started
    assert "略過已取消的：stale.mp4" in "\n".join(queued_logs(app))


def test_cancel_button_is_on_the_window_and_wired_to_cancel_all(make_app, monkeypatch):
    pressed = []
    monkeypatch.setattr(gui.App, "cancel_all", lambda self: pressed.append(self))     # 要在建立視窗之前換，按鈕建立時才綁得到

    def find(widget):
        for child in widget.winfo_children():
            if child.winfo_class() == "TButton" and str(child.cget("text")) == "取消處理":
                return child
            found = find(child)
            if found is not None:
                return found
    app = make_app()
    button = find(app.root)
    assert button is not None
    button.invoke()
    assert pressed == [app]


# ================= 錯誤回報 =================
def test_processing_errors_show_type_and_message_and_write_the_traceback(make_app, fake_jobs, tmp_path, capsys):
    app = make_app()
    app._handle_dropped([touch(tmp_path / "boom1.mp4")])
    assert wait_for(lambda: app._pending == 0)
    assert "處理「boom1.mp4」時發生錯誤：ValueError: boom 模擬" in "\n".join(queued_logs(app))
    err = capsys.readouterr().err
    assert "Traceback (most recent call last)" in err and "ValueError: boom 模擬" in err      # 完整堆疊（正式執行時會進 log 檔）
    app._enqueue({"kind": "transcribe"})                                   # 工作資料壞掉（缺欄位）
    assert wait_for(lambda: app._pending == 0)
    assert "KeyError" in "\n".join(queued_logs(app))
    app._handle_dropped([touch(tmp_path / "ok_after.mp4")])
    assert wait_for(lambda: fake_jobs.finished == ["ok_after.mp4"])        # 工作執行緒沒有因此死掉


def test_tk_callback_errors_are_reported_once_per_distinct_error(make_app):
    app = make_app()
    assert app.root.report_callback_exception == app._report_callback_exception
    app._report_callback_exception(ValueError, ValueError("介面壞了"), None)
    app._report_callback_exception(ValueError, ValueError("介面壞了"), None)
    assert "\n".join(queued_logs(app)).count("介面發生錯誤：ValueError: 介面壞了") == 1
    assert [c[0] for c in make_app.dialogs.calls if c[0] == "showerror"] == ["showerror"] and "介面壞了" in make_app.dialogs.calls[-1][2]
    app._report_callback_exception(KeyError, KeyError("另一個"), None)
    assert len([c for c in make_app.dialogs.calls if c[0] == "showerror"]) == 2


# ================= 關閉視窗 =================
def test_closing_when_idle_just_closes(make_app):
    app = make_app()
    closed = []
    app.root.destroy = lambda: closed.append(1)
    generation = JOBS.generation
    app._on_close()
    assert closed == [1] and not make_app.dialogs.calls and JOBS.generation == generation + 1


def test_closing_while_working_asks_first(make_app):
    app = make_app()
    closed = []
    app.root.destroy = lambda: closed.append(1)
    app._pending = 1
    generation = JOBS.generation
    make_app.dialogs.answers["askokcancel"] = False
    app._on_close()
    assert not closed and JOBS.generation == generation and [c[0] for c in make_app.dialogs.calls] == ["askokcancel"]
    make_app.dialogs.answers["askokcancel"] = True
    app._on_close()
    assert closed == [1] and JOBS.generation == generation + 1
    app._pending = 0


# ================= DPI =================
def fake_windows(monkeypatch, shcore_error=None, user32_error=False, name="nt"):
    calls = []

    class Shcore:
        def SetProcessDpiAwareness(self, value):
            calls.append(("shcore", value))
            if shcore_error:
                raise shcore_error
            return 0

    class User32:
        def SetProcessDPIAware(self):
            calls.append(("user32",))
            if user32_error:
                raise OSError("x")
            return 1
    monkeypatch.setitem(__import__("sys").modules, "ctypes", types.SimpleNamespace(windll=types.SimpleNamespace(shcore=Shcore(), user32=User32())))
    monkeypatch.setattr(gui, "os", types.SimpleNamespace(name=name))
    return calls


def test_dpi_awareness_on_windows(monkeypatch):
    calls = fake_windows(monkeypatch)
    gui._enable_dpi_awareness()
    assert calls == [("shcore", 1)]


def test_dpi_awareness_falls_back_on_old_windows(monkeypatch):
    calls = fake_windows(monkeypatch, shcore_error=OSError("沒有 shcore"))
    gui._enable_dpi_awareness()
    assert calls == [("shcore", 1), ("user32",)]


def test_dpi_awareness_never_crashes(monkeypatch):
    fake_windows(monkeypatch, shcore_error=AttributeError(), user32_error=True)
    gui._enable_dpi_awareness()


def test_dpi_awareness_does_nothing_off_windows(monkeypatch):
    calls = fake_windows(monkeypatch, name="posix")
    gui._enable_dpi_awareness()
    assert calls == []


def test_run_declares_dpi_awareness_before_creating_the_window(monkeypatch):
    order = []

    class Root:
        def mainloop(self):
            order.append("mainloop")
    monkeypatch.setattr(gui, "_enable_dpi_awareness", lambda: order.append("dpi"))
    monkeypatch.setattr(gui, "TkinterDnD", types.SimpleNamespace(Tk=lambda: (order.append("Tk()"), Root())[1]))
    monkeypatch.setattr(gui, "App", lambda root: order.append("App"))
    gui.run()
    assert order == ["dpi", "Tk()", "App", "mainloop"]


def test_pixel_sizes_scale_with_the_screen_dpi(make_app):
    app = make_app()
    scale = app.root.winfo_fpixels("1i") / 96.0
    assert app._px(100) == round(100 * scale)
