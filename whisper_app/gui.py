"""介面（QuizKnock 風格配色：白／紅／黑）。"""

from __future__ import annotations

import os
import queue
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD

from . import config, runtime
from .dropped import collect_dropped_files
from .jobs import JOBS, JobCancelled
from .pipeline import align_existing_srt, process_file
from .settings_store import load_settings, save_settings


class Theme:
    """集中管理配色與字型，方便日後整體調整風格。"""
    WHITE = "#FFFFFF"
    OFF_WHITE = "#F7F7F7"          # 卡片內部小分區用的淡灰白，避免整頁死白
    RED = "#E4032E"                # 主色：QuizKnock 招牌紅
    RED_DARK = "#B2021F"           # 按下 / hover 時的深紅
    RED_SOFT = "#FDE8EB"           # 極淡紅，用於提示或 hover 底色
    BLACK = "#1A1A1A"              # 主要文字／粗框線用的近黑色
    GRAY_TEXT = "#5A5A5A"

    FONT_FAMILY = "Microsoft JhengHei"
    FONT_MONO = "Consolas"

    TITLE_FONT = (FONT_FAMILY, 16, "bold")
    SUBTITLE_FONT = (FONT_FAMILY, 9)
    SECTION_FONT = (FONT_FAMILY, 11, "bold")
    BODY_FONT = (FONT_FAMILY, 10)
    BUTTON_FONT = (FONT_FAMILY, 10, "bold")
    DROP_FONT = (FONT_FAMILY, 12, "bold")
    DROP_SUB_FONT = (FONT_FAMILY, 9)
    SMALL_BUTTON_FONT = (FONT_FAMILY, 9, "bold")
    LOG_FONT = (FONT_MONO, 9)


class FlowGrid(ttk.Frame):
    """
    視窗拉寬就多排幾欄、變窄就自動換行，不再固定 4 欄。
    欄數的決定方式跟 ls 指令排檔名一樣：從最多欄開始試，每一欄的寬度取該欄
    最長的項目，總寬放得下就用這個欄數——所以既排得緊湊，同一欄也會對齊。
    """
    ITEM_PADX = 4
    ITEM_PADY = 1

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._items: list = []
        self._cols = 0
        self._width = 1
        self._pending = False
        self.bind("<Configure>", self._on_configure)

    def add(self, widget):
        """加入一個子元件（建立時 master 要指定為這個 FlowGrid）。"""
        self._items.append(widget)
        self._cols = 0  # 項目變了，強制重排
        self._schedule()

    def _on_configure(self, event):
        if event.width != self._width:
            self._width = event.width
            self._schedule()

    def _schedule(self):
        # 拖拉視窗時 <Configure> 會連續觸發很多次，合併成閒置時重排一次就好
        if not self._pending:
            self._pending = True
            self.after_idle(self._relayout)

    def _relayout(self):
        self._pending = False
        if not self._items:
            return
        widths = [w.winfo_reqwidth() + 2 * self.ITEM_PADX for w in self._items]
        cols = 1
        for n in range(len(widths), 0, -1):
            # 依序排列時第 c 欄放的是第 c, c+n, c+2n… 個項目
            if sum(max(widths[c::n]) for c in range(n)) <= self._width:
                cols = n
                break
        if cols == self._cols:
            return
        self._cols = cols
        for i, w in enumerate(self._items):
            w.grid(row=i // cols, column=i % cols, sticky="w",
                   padx=self.ITEM_PADX, pady=self.ITEM_PADY)


class AutoScrollFrame(ttk.Frame):
    """
    內容放在 self.body 裡。高度夠的時候看起來就是一般的 Frame（沒有捲軸）；
    螢幕太矮、視窗被限制高度時，右側才出現捲軸，滑鼠滾輪也能捲。
    """

    def __init__(self, master, bg: str, **kw):
        super().__init__(master, **kw)
        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0, width=1, height=1)
        self._vbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vbar.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        self.body = ttk.Frame(self._canvas, style="Root.TFrame")
        self._win = self._canvas.create_window(0, 0, window=self.body, anchor="nw")
        self._bar_shown = False
        self.body.bind("<Configure>", self._on_body_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self.bind("<Enter>", lambda e: self._bind_wheel(True))
        self.bind("<Leave>", lambda e: self._bind_wheel(False))

    def _on_body_configure(self, event=None):
        # 容器「想要」的高度 = 內容高度，空間夠時外層就會給足，不會出現捲軸
        self._canvas.configure(height=self.body.winfo_reqheight())
        self._refresh()

    def _on_canvas_configure(self, event):
        self._canvas.itemconfigure(self._win, width=event.width)
        self._refresh()

    def _refresh(self):
        content_h = self.body.winfo_reqheight()
        view_h = self._canvas.winfo_height()
        self._canvas.configure(scrollregion=(0, 0, 0, content_h))
        need = content_h > view_h + 1
        if need and not self._bar_shown:
            self._vbar.pack(side="right", fill="y")
        elif not need and self._bar_shown:
            self._vbar.pack_forget()
            self._canvas.yview_moveto(0)
        self._bar_shown = need

    def _bind_wheel(self, on: bool):
        if on:
            self.bind_all("<MouseWheel>", self._on_wheel)                              # Windows / macOS
            self.bind_all("<Button-4>", lambda e: self._scroll(-1))                    # Linux
            self.bind_all("<Button-5>", lambda e: self._scroll(1))
        else:
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.unbind_all(seq)

    def _on_wheel(self, event):
        self._scroll(-1 if event.delta > 0 else 1)

    def _scroll(self, direction: int):
        if self._bar_shown:
            self._canvas.yview_scroll(direction * 2, "units")


class App:
    # 視窗大小以 100% 縮放（96 DPI）設計；螢幕不夠大時會自動縮小，保證整個視窗都在螢幕內
    WIN_W, WIN_H = 1120, 680
    MIN_W, MIN_H = 860, 560
    LEFT_W = 520        # 左欄（設定／Hotwords／拖放區）寬度，視窗拉寬時多出來的空間都給右邊日誌
    SCREEN_RESERVE = 110  # 螢幕高度要預留給工作列與視窗標題列的空間

    def __init__(self, root):
        self.root = root
        self._settings = load_settings()  # 上次記住的勾選狀態；沒有就全部用預設值
        self._save_pending = False
        self._save_failed = False
        root.title("Whisper 字幕產生器")
        root.configure(bg=Theme.WHITE)
        self._fit_window()

        self._setup_style()
        self._build_header()

        # 左右兩欄：左邊是操作區（寬度固定），右邊是進度與日誌（吃掉剩下的寬度，而且永遠是全高）
        body = ttk.Frame(root, style="Root.TFrame")
        body.pack(fill="both", expand=True, padx=14, pady=12)
        body.columnconfigure(0, minsize=self._px(self.LEFT_W))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="Root.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        # 拖放區先佔好左欄底部（空間不夠時才不會被擠掉），多出來的高度也都給它；
        # 設定＋Hotwords 放在上面，螢幕太矮時這一塊才出現捲軸
        self._build_drop_zone(left)
        controls = AutoScrollFrame(left, bg=Theme.WHITE, style="Root.TFrame")
        controls.pack(side="top", fill="x")
        self._build_settings(controls.body)
        self._build_hotwords(controls.body)
        self._build_log_panel(body)
        for var in [self.use_sep_var, self.use_align_var, self.use_split_var, self.add_blank_var,
                    *self.hotword_vars.values()]:
            var.trace_add("write", self._on_setting_changed)  # 建好之後才掛，建立時的預設值不算「改動」

        # --- 背景處理：佇列 + 工作執行緒 ---
        self.file_queue = queue.Queue()
        self.ui_queue = queue.Queue()
        self._pending = 0  # 排隊中＋處理中的工作數，關閉視窗時用來判斷要不要先問
        self._pending_lock = threading.Lock()
        self._reported_errors: set[str] = set()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.report_callback_exception = self._report_callback_exception
        threading.Thread(target=self.worker_loop, daemon=True).start()
        self.root.after(100, self.poll_ui_queue)
        for note in runtime.STARTUP_NOTES:
            self.log(note)

    # ---- 視窗大小：依螢幕大小決定，不會超出螢幕 ----
    def _px(self, n: float) -> int:
        """設計尺寸（96 DPI）→ 實際像素。Tk 以高 DPI 模式執行時會跟著放大。"""
        return int(round(n * self.root.winfo_fpixels("1i") / 96.0))

    def _fit_window(self):
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        reserve = self._px(self.SCREEN_RESERVE)
        w = min(self._px(self.WIN_W), sw - self._px(40))
        h = min(self._px(self.WIN_H), sh - reserve)
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - reserve - h) // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.minsize(min(self._px(self.MIN_W), w), min(self._px(self.MIN_H), h))

    # ---- 左欄：設定 ----
    def _build_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="設定（拖檔案進來之前先選好）", style="Card.TLabelframe")
        frame.pack(fill="x")
        inner = ttk.Frame(frame, style="Card.TFrame")
        inner.pack(fill="x", padx=10, pady=(2, 8))

        self.use_sep_var = tk.BooleanVar(value=bool(self._settings.get("use_sep", True)))
        self.use_align_var = tk.BooleanVar(value=bool(self._settings.get("use_align", True)))
        self.use_split_var = tk.BooleanVar(value=bool(self._settings.get("use_split", True)))
        self.add_blank_var = tk.BooleanVar(value=bool(self._settings.get("add_blank", True)))
        for var, text in [
            (self.use_sep_var, "先用 Demucs 分離人聲"),
            (self.use_align_var, "用 wav2vec2 強制對齊，校正時間戳"),
            (self.use_split_var, "自動拆分塞了好幾句的字幕"),
            (self.add_blank_var, "開頭加一條空白字幕（方便剪輯軟體對軸）"),
        ]:
            ttk.Checkbutton(
                inner, text=text, variable=var, style="Card.TCheckbutton",
            ).pack(anchor="w", pady=1)

    # ---- 左欄：Hotwords（這次影片有出現的人名/關鍵字才勾，不相關的反而會干擾辨識）----
    def _build_hotwords(self, parent):
        frame = ttk.LabelFrame(parent, style="Card.TLabelframe")
        frame.pack(fill="x", pady=(10, 0))
        # 標題文字跟「全選／全不選」放在同一行（當作外框的標題），省下一整列的高度
        title = ttk.Frame(frame, style="Card.TFrame")
        ttk.Label(title, text="Hotwords（勾選這次會出現的詞）", style="CardTitle.TLabel").pack(side="left")
        ttk.Button(
            title, text="全選", style="Small.Red.TButton",
            command=lambda: self._set_all_hotwords(True),
        ).pack(side="left", padx=(10, 0))
        ttk.Button(
            title, text="全不選", style="Small.RedOutline.TButton",
            command=lambda: self._set_all_hotwords(False),
        ).pack(side="left", padx=(6, 0))
        frame.configure(labelwidget=title)

        self.hotword_vars: dict[str, tk.BooleanVar] = {}
        self._hotword_row(frame, "人名", config.HOTWORD_NAME_CANDIDATES, pady=(6, 2))
        self._hotword_row(frame, "通用詞", config.HOTWORD_COMMON_CANDIDATES, pady=(6, 8))

    def _hotword_row(self, parent, tag: str, words: list[str], pady):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", padx=10, pady=pady)
        # 兩個分類標籤設成相同最小寬度，右邊的勾選框起點才會對齊
        ttk.Label(
            row, text=tag, style="Tag.TLabel", width=-6, anchor="center",
        ).pack(side="left", anchor="n", pady=(2, 0))
        grid = FlowGrid(row, style="Card.TFrame")
        grid.pack(side="left", fill="x", expand=True, padx=(8, 0))
        saved = self._settings.get("hotwords")
        saved = saved if isinstance(saved, dict) else {}
        for word in words:
            # 第一次開啟：人名每部影片不同，預設不勾；通用詞預設勾。之後沿用上次的選擇
            var = tk.BooleanVar(value=bool(saved.get(word, word in config.HOTWORD_COMMON_CANDIDATES)))
            self.hotword_vars[word] = var
            grid.add(ttk.Checkbutton(grid, text=word, variable=var, style="Card.TCheckbutton"))

    # ---- 左欄：拖放區（填滿左欄剩下的高度，視窗拉高時目標也跟著變大）----
    def _build_drop_zone(self, parent):
        self.drop_zone = tk.Frame(
            parent, bg=Theme.WHITE, height=self._px(110),
            highlightthickness=self._px(3), highlightbackground=Theme.RED, highlightcolor=Theme.RED,
            cursor="hand2",
        )
        self.drop_zone.pack(side="bottom", fill="both", expand=True, pady=(10, 0))
        inner = tk.Frame(self.drop_zone, bg=Theme.WHITE)
        inner.place(relx=0.5, rely=0.5, anchor="center")
        self._drop_main = tk.Label(
            inner, text="⬇  把影片或音訊拖到這裡開始轉錄  ⬇",
            bg=Theme.WHITE, fg=Theme.BLACK, font=Theme.DROP_FONT,
        )
        self._drop_main.pack()
        self._drop_sub = tk.Label(
            inner,
            text="可一次拖多個檔案，會自動排隊依序處理\n只想重新對齊現有字幕：把 .srt 和影片一起拖進來",
            bg=Theme.WHITE, fg=Theme.GRAY_TEXT, font=Theme.DROP_SUB_FONT, justify="center",
        )
        self._drop_sub.pack(pady=(4, 0))

        # 檔案放在框內任何地方（包括文字上）都要收得到
        self._drop_widgets = [self.drop_zone, inner, self._drop_main, self._drop_sub]
        for w in self._drop_widgets:
            w.drop_target_register(DND_FILES)
            w.dnd_bind("<<Drop>>", self.on_drop)
        # 純滑鼠 hover 也給點回饋，讓拖放區感覺「活著」
        self.drop_zone.bind("<Enter>", lambda e: self._paint_drop(Theme.RED_SOFT))
        self.drop_zone.bind("<Leave>", self._on_drop_zone_leave)

    def _paint_drop(self, bg: str, flash: bool = False):
        for w in self._drop_widgets:
            w.configure(bg=bg)
        self._drop_main.configure(fg=Theme.WHITE if flash else Theme.BLACK)
        self._drop_sub.configure(fg=Theme.WHITE if flash else Theme.GRAY_TEXT)

    def _on_drop_zone_leave(self, event):
        # 滑鼠移到框內的文字上也會觸發 <Leave>，只有真的離開整個框才恢復白底
        under = self.root.winfo_containing(event.x_root, event.y_root)
        if under is None or not str(under).startswith(str(self.drop_zone)):
            self._paint_drop(Theme.WHITE)

    # ---- 右欄：進度與日誌（全高）----
    def _build_log_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="進度與日誌", style="Card.TLabelframe")
        panel.grid(row=0, column=1, sticky="nsew")
        inner = ttk.Frame(panel, style="Card.TFrame")
        inner.pack(fill="both", expand=True, padx=10, pady=(2, 10))

        status_row = ttk.Frame(inner, style="Card.TFrame")
        status_row.pack(fill="x")
        ttk.Label(
            status_row, text="●", foreground=Theme.RED, background=Theme.WHITE,
            font=(Theme.FONT_FAMILY, 10),
        ).pack(side="left")
        self.status_var = tk.StringVar(value=" 等待拖入檔案…")
        ttk.Label(
            status_row, textvariable=self.status_var, style="Status.TLabel",
        ).pack(side="left")
        ttk.Button(
            status_row, text="取消處理", style="Small.RedOutline.TButton", command=self.cancel_all,
        ).pack(side="right")

        self.progress = ttk.Progressbar(
            inner, mode="determinate", maximum=100, style="Red.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", pady=(4, 8))

        log_wrap = tk.Frame(inner, bg=Theme.BLACK, bd=0, highlightthickness=0)
        log_wrap.pack(fill="both", expand=True)
        self.log_box = scrolledtext.ScrolledText(
            log_wrap, height=10, width=40, state="disabled",
            bg=Theme.OFF_WHITE, fg=Theme.BLACK, insertbackground=Theme.BLACK,
            font=Theme.LOG_FONT, relief="flat", bd=0,
            padx=8, pady=6,
        )
        self.log_box.pack(fill="both", expand=True, padx=2, pady=2)

    # ---- 佈景主題設定：統一設定 ttk 樣式，讓白／紅／黑配色套用到所有元件 ----
    def _setup_style(self):
        style = ttk.Style(self.root)
        # 'clam' 是純 Tk 主題（非系統原生外觀），才能讓自訂顏色在 Windows 上生效
        style.theme_use("clam")

        style.configure("Root.TFrame", background=Theme.WHITE)
        style.configure("Card.TFrame", background=Theme.WHITE)
        style.configure("TFrame", background=Theme.WHITE)

        style.configure(
            "Card.TLabelframe", background=Theme.WHITE,
            bordercolor=Theme.BLACK, borderwidth=self._px(2), relief="solid",
        )
        style.configure(
            "Card.TLabelframe.Label", background=Theme.WHITE,
            foreground=Theme.RED, font=Theme.SECTION_FONT,
        )

        style.configure("Card.TLabel", background=Theme.WHITE, foreground=Theme.BLACK, font=Theme.BODY_FONT)
        # 自訂外框標題（Hotwords 那一列）用的文字，外觀跟一般外框標題一致
        style.configure("CardTitle.TLabel", background=Theme.WHITE, foreground=Theme.RED, font=Theme.SECTION_FONT)
        style.configure("Status.TLabel", background=Theme.WHITE, foreground=Theme.BLACK, font=Theme.BUTTON_FONT)
        # 「人名 / 通用詞」小標籤：黑底白字的小徽章，呼應黑／紅／白三色
        style.configure(
            "Tag.TLabel", background=Theme.BLACK, foreground=Theme.WHITE,
            font=(Theme.FONT_FAMILY, 9, "bold"), padding=(self._px(6), self._px(2)),
        )

        style.configure(
            "TCheckbutton", background=Theme.WHITE, foreground=Theme.BLACK, font=Theme.BODY_FONT,
        )
        style.configure(
            "Card.TCheckbutton", background=Theme.WHITE, foreground=Theme.BLACK, font=Theme.BODY_FONT,
            indicatorsize=self._px(10),  # 勾選框的大小是像素，螢幕縮放大時要跟著放大，不然會比字小很多
        )
        style.map(
            "Card.TCheckbutton",
            indicatorcolor=[("selected", Theme.RED), ("!selected", Theme.WHITE)],
            background=[("active", Theme.WHITE)],
            foreground=[("active", Theme.RED)],
        )

        # 主要按鈕：紅底白字
        style.configure(
            "Red.TButton", background=Theme.RED, foreground=Theme.WHITE,
            font=Theme.BUTTON_FONT, borderwidth=0, focusthickness=0,
            padding=(self._px(14), self._px(6)),
        )
        style.map(
            "Red.TButton",
            background=[("active", Theme.RED_DARK), ("pressed", Theme.RED_DARK)],
            foreground=[("active", Theme.WHITE)],
        )

        # 次要按鈕：白底紅字紅框，跟主按鈕做出層級區分
        style.configure(
            "RedOutline.TButton", background=Theme.WHITE, foreground=Theme.RED,
            font=Theme.BUTTON_FONT, borderwidth=self._px(1.5), bordercolor=Theme.RED,
            padding=(self._px(14), self._px(6)),
        )
        style.map(
            "RedOutline.TButton",
            background=[("active", Theme.RED_SOFT), ("pressed", Theme.RED_SOFT)],
            foreground=[("active", Theme.RED)],
        )

        # 小尺寸按鈕（放在外框標題列）：「Small.Red.TButton」會沿用 Red.TButton 的顏色，只改大小
        small_padding = (self._px(10), self._px(1))
        style.configure("Small.Red.TButton", font=Theme.SMALL_BUTTON_FONT, padding=small_padding)
        style.configure("Small.RedOutline.TButton", font=Theme.SMALL_BUTTON_FONT, padding=small_padding)

        # 進度條：紅色前景、淡灰底、黑色外框
        style.configure(
            "Red.Horizontal.TProgressbar",
            troughcolor=Theme.OFF_WHITE, background=Theme.RED,
            bordercolor=Theme.BLACK, lightcolor=Theme.RED, darkcolor=Theme.RED,
            thickness=self._px(14),
        )

    # ---- 頂部標題橫幅：紅底白字 + 黑色粗分隔線，營造 QuizKnock 節目片頭感 ----
    def _build_header(self):
        banner = tk.Frame(self.root, bg=Theme.RED)
        banner.pack(fill="x")

        inner = tk.Frame(banner, bg=Theme.RED)
        inner.pack(fill="x", padx=14, pady=(8, 7))

        mark = tk.Label(
            inner, text="▶", bg=Theme.BLACK, fg=Theme.WHITE,
            font=(Theme.FONT_FAMILY, 16, "bold"), width=2, height=1,
        )
        mark.pack(side="left", padx=(0, 10))

        text_col = tk.Frame(inner, bg=Theme.RED)
        text_col.pack(side="left", fill="x", expand=True)
        tk.Label(
            text_col, text="QuizKnock專用字幕產生器", bg=Theme.RED, fg=Theme.WHITE,
            font=Theme.TITLE_FONT, anchor="w",
        ).pack(fill="x")
        tk.Label(
            text_col, text=f"拖放檔案即可自動轉錄・Whisper {config.WHISPER_MODEL}",
            bg=Theme.RED, fg=Theme.WHITE, font=Theme.SUBTITLE_FONT, anchor="w",
        ).pack(fill="x")

        tk.Frame(self.root, bg=Theme.BLACK, height=self._px(3)).pack(fill="x")

    # ---- 給背景執行緒呼叫，把更新丟回主執行緒（不可直接從子執行緒操作 Tk 元件）----
    def log(self, msg: str):
        self.ui_queue.put(("log", msg))

    def set_progress(self, frac: float):
        self.ui_queue.put(("progress", frac))

    def set_status(self, msg: str):
        self.ui_queue.put(("status", msg))

    def poll_ui_queue(self):
        lines = []
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "log":
                    lines.append(str(payload))
                elif kind == "progress":
                    self.progress["value"] = float(payload) * 100
                elif kind == "status":
                    self.status_var.set(" " + str(payload))
        except queue.Empty:
            pass
        finally:
            if lines:
                self._append_log(lines)  # 這一輪收到的日誌一次寫進去，比一行一行寫快
            # 就算處理某一筆時出錯，也要讓計時器繼續跑，不然畫面之後就不會更新了
            self.root.after(100, self.poll_ui_queue)

    def _append_log(self, lines: list[str]) -> None:
        """
        把日誌寫進視窗。只有本來就捲在最底下時才跟著捲到最新；往上捲去看舊訊息時不會被拉回來。
        超過 LOG_MAX_LINES 行就從最舊的開始刪（完整內容還在 log 檔裡）。
        """
        box = self.log_box
        at_bottom = box.yview()[1] >= 0.999
        box.configure(state="normal")
        box.insert("end", "\n".join(lines) + "\n")
        excess = int(box.index("end-1c").split(".")[0]) - 1 - config.LOG_MAX_LINES
        if excess > 0:
            box.delete("1.0", f"{excess + 1}.0")
        box.configure(state="disabled")
        if at_bottom:
            box.see("end")

    def _set_all_hotwords(self, checked: bool):
        for var in self.hotword_vars.values():
            var.set(checked)

    # ---- 記住設定：勾選一有變動就排一次存檔（連續改很多個只存一次）----
    def _on_setting_changed(self, *_):
        if not self._save_pending:
            self._save_pending = True
            self.root.after(500, self._save_settings)

    def _save_settings(self):
        self._save_pending = False
        data = {
            "version": 1,
            "use_sep": self.use_sep_var.get(), "use_align": self.use_align_var.get(),
            "use_split": self.use_split_var.get(), "add_blank": self.add_blank_var.get(),
            "hotwords": {word: var.get() for word, var in self.hotword_vars.items()},
        }
        if not save_settings(data) and not self._save_failed:
            self._save_failed = True  # 只提醒一次
            self.log(f">>> 注意：無法儲存設定到 {config.SETTINGS_FILE}（下次開啟會回到預設值）")

    # ---- 拖放事件：把檔案丟進佇列，並給一次紅色閃爍當作「已接收」的視覺回饋 ----
    def on_drop(self, event):
        paths = list(self.root.tk.splitlist(event.data))
        self._paint_drop(Theme.RED, flash=True)
        self.root.after(180, lambda: self._paint_drop(Theme.WHITE))
        # 拖放事件結束後才處理（可能會跳出選檔視窗，不能卡住拖放來源的檔案總管）
        self.root.after(50, lambda: self._handle_dropped(paths))

    def _handle_dropped(self, paths: list[str]):
        use_sep = self.use_sep_var.get()
        add_blank = self.add_blank_var.get()
        media_files, srt_files, skipped = collect_dropped_files(paths)
        if skipped:
            shown = "、".join(skipped[:5]) + (f" 等 {len(skipped)} 個" if len(skipped) > 5 else "")
            self.log(f">>> 略過 {len(skipped)} 個不是影片／音訊的項目：{shown}")
        if not media_files and not srt_files:
            self.log(">>> 沒有可以處理的影片或音訊檔案")
            return

        # 有 .srt → 只對齊模式：拖進來的影片只拿來配對，不會重新轉錄
        if srt_files:
            used_media = set()
            for srt in srt_files:
                media = self._match_media_for_srt(srt, media_files)
                if media is None:
                    media = filedialog.askopenfilename(
                        parent=self.root,
                        title=f"選擇「{Path(srt).name}」對應的影片或音訊",
                        filetypes=[
                            ("影片／音訊", " ".join("*" + ext for ext in sorted(config.MEDIA_EXTENSIONS))),
                            ("所有檔案", "*.*"),
                        ],
                    )
                if not media:
                    self.log(f">>> 略過「{Path(srt).name}」：沒有選擇對應的影片")
                    continue
                used_media.add(media)
                self._enqueue({"kind": "align_only", "srt": srt, "media": media,
                               "add_blank": add_blank})
                self.log(f">>> 已加入佇列（只對齊）：{Path(srt).name}  ⇄  {Path(media).name}")
            for m in media_files:
                if m not in used_media:
                    self.log(f">>> 注意：「{Path(m).name}」沒有配對到任何 .srt，這次不處理"
                             f"（要轉錄請單獨拖進來）")
            return

        # 沒有 .srt → 原本的轉錄流程
        use_align = self.use_align_var.get()
        use_split = self.use_split_var.get()
        selected_hotwords = [w for w, var in self.hotword_vars.items() if var.get()]
        if len(media_files) > config.MANY_FILES_CONFIRM and not messagebox.askyesno(
                "Whisper 字幕產生器", f"這次要加入 {len(media_files)} 個檔案，全部都處理嗎？", parent=self.root):
            self.log(f">>> 已取消加入（{len(media_files)} 個檔案）")
            return
        for p in media_files:
            self._enqueue({
                "kind": "transcribe", "path": p, "use_sep": use_sep, "use_align": use_align,
                "use_split": use_split, "hotwords": selected_hotwords, "add_blank": add_blank,
            })
        self.log(f">>> 已加入佇列：{len(media_files)} 個檔案"
                 f"（hotwords：{'、'.join(selected_hotwords) if selected_hotwords else '無'}）")

    @staticmethod
    def _match_media_for_srt(srt: str, media_files: list[str]):
        """
        幫 .srt 找對應的影片：只拖了一個影片就直接用它；
        多個影片時，找檔名是 srt 檔名開頭的那個（例如 abc.mp4 ⇄ abc_large-v3_sep.srt）。
        """
        if len(media_files) == 1:
            return media_files[0]
        srt_stem = Path(srt).stem
        candidates = [m for m in media_files if srt_stem.startswith(Path(m).stem)]
        if not candidates:
            return None
        return max(candidates, key=lambda m: len(Path(m).stem))

    # ---- 背景執行緒：依序處理佇列中的檔案 ----
    def worker_loop(self):
        while True:
            job = self.file_queue.get()
            name = "（未知的工作）"
            try:
                is_align_only = job["kind"] == "align_only"
                name = Path(job["srt"] if is_align_only else job["path"]).name
                if JOBS.is_cancelled(job["gen"]):
                    self.log(f">>> 略過已取消的：{name}")
                    continue
                JOBS.begin(job["gen"])
                self.set_status(f"{'對齊中' if is_align_only else '處理中'}：{name}")
                self.set_progress(0)
                if is_align_only:
                    align_existing_srt(job["srt"], job["media"],
                                       self.log, self.set_progress,
                                       add_blank=job["add_blank"])
                else:
                    process_file(job["path"], job["use_sep"], job["use_align"],
                                 job["use_split"], job["hotwords"],
                                 self.log, self.set_progress,
                                 add_blank=job["add_blank"])
                self.set_status("完成，等待下一個檔案…")
            except JobCancelled:
                self.log(f">>> 已取消：{name}")
                self.set_status("已取消")
                self.set_progress(0)
            except Exception as e:
                traceback.print_exc()  # 完整的錯誤堆疊寫進 log 檔
                where = f"（詳細內容在 {runtime.LOG_FILE}）" if runtime.LOG_FILE else ""
                self.log(f"!!! 處理「{name}」時發生錯誤：{type(e).__name__}: {e}{where}")
                self.set_status("發生錯誤，請查看上方日誌")
            finally:
                self._job_finished()

    # ---- 排隊數量、取消、關閉視窗 ----
    def _enqueue(self, job: dict) -> None:
        job["gen"] = JOBS.generation  # 之後按了取消，這個 generation 之前排進來的工作都會作廢
        with self._pending_lock:
            self._pending += 1
        self.file_queue.put(job)

    def _job_finished(self) -> None:
        with self._pending_lock:
            self._pending -= 1

    def _is_working(self) -> bool:
        with self._pending_lock:
            return self._pending > 0

    def cancel_all(self):
        """取消：停掉目前處理的檔案（連同背景的 Demucs／對齊程序），並清掉排隊中的。"""
        if not self._is_working():
            self.log(">>> 目前沒有處理中或排隊中的檔案")
            return
        JOBS.cancel()  # 先作廢，之後不管 worker 取到哪個舊工作都會被略過
        dropped = 0
        try:
            while True:
                self.file_queue.get_nowait()
                dropped += 1
        except queue.Empty:
            pass
        with self._pending_lock:
            self._pending -= dropped
            running = self._pending > 0
        self.log(">>> 已要求取消" + (f"，並清除排隊中的 {dropped} 個檔案" if dropped else ""))
        if running:
            self.set_status("正在取消…")
        else:
            self.set_status("已取消，等待拖入檔案…")
            self.set_progress(0)

    def _on_close(self):
        if self._is_working() and not messagebox.askokcancel(
                "Whisper 字幕產生器",
                "還有檔案在處理中（或排隊中），現在關閉會中止它們。\n\n確定要關閉嗎？",
                parent=self.root):
            return
        self._save_settings()
        JOBS.cancel()
        self.root.destroy()

    def _report_callback_exception(self, exc, val, tb):
        """Tk 事件處理函式（拖放、按鈕、計時器）裡沒接住的例外：寫進 log 檔，並提示使用者。"""
        traceback.print_exception(exc, val, tb)
        key = f"{exc.__name__}: {val}"
        if key in self._reported_errors:  # 同一個錯誤（例如計時器每 0.1 秒一次）只提示一次
            return
        self._reported_errors.add(key)
        where = f"\n\n詳細錯誤已寫入：\n{runtime.LOG_FILE}" if runtime.LOG_FILE else ""
        self.log(f"!!! 介面發生錯誤：{key}")
        try:
            messagebox.showerror("Whisper 字幕產生器 - 發生錯誤", f"{key}{where}", parent=self.root)
        except Exception:
            pass


def _enable_dpi_awareness() -> None:
    """
    Windows：告訴系統這個程式自己會處理螢幕縮放。不宣告的話，螢幕縮放大於 100% 時
    系統會把整個視窗當成圖片放大，字會糊掉，而且 _px() 量到的永遠是 96 DPI。
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # 系統層級的 DPI 感知（Python 的 IDLE 也是這樣做）
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()       # 舊版 Windows 沒有 shcore
    except Exception:
        pass


def run() -> None:
    """建立視窗並進入主迴圈。"""
    _enable_dpi_awareness()  # 要在建立 Tk 視窗之前
    root = TkinterDnD.Tk()
    App(root)
    root.mainloop()
