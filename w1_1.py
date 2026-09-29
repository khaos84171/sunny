"""
Whisper 字幕產生器（拖放視窗版／可雙擊啟動）
— 介面採 QuizKnock 風格配色（白／紅／黑）—

用法：
    1. 安裝額外套件：pip install tkinterdnd2 transformers janome
       （transformers 是「時間戳對齊」功能用的；沒裝也能跑，只是會自動略過對齊）
       （janome 是日文斷詞，讓「自動拆分」只在詞與詞之間切，不會把「熊本」切成「熊／本」；
         沒裝也能跑，會改用比較粗略的判斷規則）
    2. 平常用「字幕產生器.vbs」雙擊開啟即可（不會跳出黑色 cmd 視窗）。
       也可以直接執行 python w1_1.py 來測試。
    3. 視窗開啟後，先確認下方的設定（要不要先做人聲分離、對齊、拆分），
       再把影片或音訊檔案拖到拖放區，會自動開始轉錄。輸出的 .srt 會用
       原始檔名加上「_large-v3_是否分離」當後綴，不會互相覆蓋，也不用每次
       改腳本裡的路徑。轉錄固定使用 Whisper large-v3。
    4. 可以一次拖多個檔案，會自動排隊依序處理。
    5. 用雙擊啟動時沒有終端機視窗可以看錯誤訊息，所有錯誤都會寫進腳本
       同資料夾下的 whisper_app.log（那個資料夾不能寫入時改放在系統暫存資料夾），
       啟動失敗也會跳出訊息框提示。
    6. 勾選「時間戳對齊」時，Whisper 轉錄完後會再用日文 wav2vec2 模型做
       CTC 強制對齊，把每條字幕的起訖時間校正到實際開口／收尾的位置。
       對齊模型第一次使用會自動從 Hugging Face 下載（約 1.2 GB）。
       對齊是由同資料夾的 align_worker.py 在獨立程序裡執行（避免 PyTorch 與
       faster-whisper 的 cuDNN 版本衝突），請務必把兩個檔案放在一起。
    7. 只想重新對齊現有字幕、不重跑 Whisper 時：把 .srt 和對應的影片「一起」
       拖進拖放區即可（例如手動改過字幕文字之後）。只拖 .srt 的話會跳出視窗
       讓你選對應的影片。輸出為原檔名加上 _align，不會覆蓋原本的 .srt。
    8. 勾選「自動拆分」時，Whisper 會多輸出每個詞的時間（word_timestamps），
       再把一條裡塞了好幾句的字幕，依「句尾標點 → 詞與詞之間的停頓 → 長度上限」
       拆成多條。有勾「時間戳對齊」時，會先拿 Whisper 原本的整段去對齊（文字長
       對得穩），align_worker.py 再傳回 wav2vec2 算出的每個詞的時間，用這個時間
       判斷停頓、決定在哪裡拆——比 Whisper 自己估的詞時間準很多。沒勾對齊時就用
       Whisper 的詞時間來拆。開啟後轉錄會稍微慢一點。門檻在下方「字幕拆分」設定區。
    9. 勾選「在最前面加一條空白字幕」時，輸出的 .srt 第一條會是從 00:00:00,000 開始、
       到第一句字幕出現為止的空白字幕。有些剪輯軟體匯入 SRT 時會把第一條字幕放在
       播放頭的位置，有這條空白字幕墊在 0 秒，把播放頭放在影片開頭再匯入就能對齊。
       只對齊模式也適用；輸入的 .srt 裡原本就有的空白字幕會先拿掉再重新加，不會變兩條。
   10. 需要 Python 3.9 以上。輸出資料夾（output_dir）和人聲分離的暫存資料夾（separation_work_dir）
       預設在 D:\桌面\whisper 底下；建立不起來（例如這台電腦沒有 D 槽）時，會自動改用腳本旁邊的
       text／separated 資料夾，實際用到哪裡會寫在視窗開啟後日誌的第一行。
   11. 人聲分離用獨立程序跑 Demucs（不會跳出黑色視窗，進度會顯示在進度條和日誌裡）。結果存在
       separation_work_dir 底下，同一個檔案（大小與修改時間沒變）再處理時直接重用。
   12. 右邊的「取消處理」會停掉目前的檔案（連同背景的 Demucs／對齊程序）並清掉排隊中的；
       處理中直接關閉視窗會先詢問，確定關閉時一樣會把背景程序停掉。
       也可以把資料夾拖進來：會把裡面（含子資料夾）的影片／音訊依檔名加入佇列，
       不是影片／音訊的檔案會略過並在日誌說明。認得的副檔名在 MEDIA_EXTENSIONS。
"""

from __future__ import annotations   # 讓 int | None、list[str] 這類標註在 Python 3.9 也能載入

import sys
import os
import tempfile
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 啟動時發生、值得讓使用者知道的事（例如改用了備用資料夾），視窗開起來後寫進日誌
STARTUP_NOTES: list[str] = []


def _open_log_file():
    """log 檔預設放在腳本旁邊；那個資料夾不能寫入（例如裝在 Program Files）就改放到暫存資料夾。"""
    for folder in (BASE_DIR, Path(tempfile.gettempdir())):
        path = folder / "whisper_app.log"
        try:
            return path, open(path, "a", encoding="utf-8", buffering=1)
        except OSError:
            continue
    return None, open(os.devnull, "w", encoding="utf-8")


# 用 pythonw 雙擊啟動時沒有終端機，sys.stdout / sys.stderr 會是 None，
# 任何 print 或例外訊息原本會直接消失甚至報錯，所以先一律導向同一個 log 檔。
LOG_FILE, _log_stream = _open_log_file()
sys.stdout = _log_stream
sys.stderr = _log_stream
if LOG_FILE is None:
    STARTUP_NOTES.append("!!! 找不到可以寫入的位置，這次不會留下 whisper_app.log")
elif LOG_FILE.parent != BASE_DIR:
    STARTUP_NOTES.append(f"注意：腳本資料夾不能寫入，日誌檔改放在 {LOG_FILE}")


def _show_fatal_error(message: str):
    """啟動失敗或執行中發生未捕捉例外時，跳出訊息框告知使用者。"""
    traceback.print_exc()
    title = "Whisper 字幕產生器 - 發生錯誤"
    text = message + (f"\n\n詳細錯誤已寫入：\n{LOG_FILE}" if LOG_FILE else "")
    try:
        import tkinter as _tk
        from tkinter import messagebox as _mb
        _root = _tk.Tk()
        _root.withdraw()
        _mb.showerror(title, text)
        _root.destroy()
    except Exception:
        # tkinter 本身載入失敗（例如安裝 Python 時沒有勾 tcl/tk）：改用 Windows 內建的訊息框
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, text, title, 0x10)
        except Exception:
            pass


try:
    import logging
    import atexit
    import codecs
    import hashlib
    import importlib.util
    import shutil
    import subprocess
    import threading
    import queue
    import re
    import json
    import collections
    import unicodedata

    import numpy as np

    import tkinter as tk
    from tkinter import ttk, scrolledtext, filedialog, messagebox

    from tkinterdnd2 import TkinterDnD, DND_FILES

    # 處理 OpenMP 衝突
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    from faster_whisper import WhisperModel, decode_audio
except Exception:
    _show_fatal_error("缺少必要套件或載入失敗（例如 tkinterdnd2 / faster_whisper 沒裝好）。")
    raise

# =========================================================
# === 固定設定（不常變動，需要的話可直接改這裡）===
# =========================================================
SEPARATION_DEVICE = "cuda"        # 與 Whisper 共用 GPU，注意顯存是否足夠
SEPARATION_MODEL = "htdemucs"     # htdemucs_ft 是 4 個模型的組合，記憶體需求約 4 倍，容易爆記憶體
                                   # 若記憶體充足想換回更高品質版本，可改成 "htdemucs_ft"
SEPARATION_SEGMENT = 7            # htdemucs 是 Transformer 架構，硬性上限 7.8 秒，不能設更大
                                   # 只能往下調（例如 5）來進一步省記憶體；設 None 則不加此參數
SEPARATION_PROGRESS_SHARE = 0.25  # 有勾人聲分離時，分離佔進度條的前多少比例（轉錄、對齊縮進剩下的部分）

# 偏好的資料夾。建立不起來（例如這台電腦沒有 D 槽）會自動改用腳本旁邊的 text／separated，
# 實際用到的位置會寫進視窗開啟後日誌的第一行。
output_dir = r"D:\桌面\whisper\text"
separation_work_dir = r"D:\桌面\whisper\separated"

# 拖放進來的影片／音訊只接受這些副檔名（拖資料夾進來時，只會挑出裡面符合的）；沒列到的格式可自己加進來
MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".ts", ".mts", ".m2ts", ".flv", ".wmv", ".m4v",
    ".mpg", ".mpeg", ".3gp", ".ogv",
    ".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif",
}
MANY_FILES_CONFIRM = 30   # 一次要加入超過這麼多個檔案（例如不小心拖了整個資料夾）時，先問一下

# === Whisper 設定（固定使用 large-v3）===
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
# 轉錄參數集中放這裡，要微調不用去翻 process_file
TRANSCRIBE_OPTIONS = dict(
    language="ja",
    beam_size=5,
    best_of=5,
    vad_filter=True,
    vad_parameters=dict(
        threshold=0.2,
        min_silence_duration_ms=400,
        speech_pad_ms=300,
    ),
    # large-v3 開著這個很容易把前面的幻覺或重複一路帶下去，關掉比較穩
    condition_on_previous_text=False,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    compression_ratio_threshold=2.4,   # 同一句一直重複（壓縮率太高）就換溫度重跑
    log_prob_threshold=-1.5,
    no_speech_threshold=0.8,
)
# large-v3 的日文輸出常常完全沒有標點，會讓「依句尾標點拆分」派不上用場。
# hotwords 會被放進每一個 30 秒視窗的提示裡，Whisper 會模仿提示的書寫風格，
# 所以把 hotwords 用「、」串起來、最後加「。」，讓它比較願意輸出日文標點。
# 如果發現反而變差（例如字幕出現奇怪的「、」），改成 False 就會回到原本的「, 」串法。
HOTWORDS_JA_PUNCTUATION = True

# hotwords 候選清單（用於前端勾選）：每部影片出現的人不同，hotwords 塞太多
# 反而會干擾 Whisper 辨識，所以改成每次依實際出場人物勾選要用哪些詞。
# 分成「人名」跟「通用詞」兩組只是方便介面分區顯示，程式邏輯上一視同仁。
HOTWORD_NAME_CANDIDATES = [
    "伊沢拓司", "ふくら", "河村拓哉", "niceguy須貝", "須貝駿貴",
    "山本祥彰", "鶴崎修功", "東もん", "もん", "もんちゃん",
    "東ごん", "ごん", "ごんちゃん", "こうちゃん", "乾",
]
HOTWORD_COMMON_CANDIDATES = [
    "クイズ", "クイズノック", "QuizKnock", "クイズ王", "東大", "誤答", "正解", "押し", "早押し", "ボタン", "GameKnack", "QuizKnockと学ぼう",
]

# === 時間戳對齊（wav2vec2 CTC 強制對齊）設定 ===
# Whisper 的片段時間常常會早開始／晚結束，或整句偏移 0.5～1 秒。
# 這裡用日文 wav2vec2 聲學模型，把 Whisper 已經辨識出的文字「對」回音訊上，
# 取第一個字與最後一個字實際發音的位置當作字幕起訖點（與 WhisperX 同樣的做法）。
ALIGN_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
ALIGN_DEVICE = "cuda"
# 對齊分兩輪：第一輪每句只在 Whisper 時間附近找（穩）；只有第一輪看起來失敗的句子
# 才在第二輪擴大範圍重找（範圍會被夾在前後對好的句子之間）。詳見 align_worker.py 開頭說明。
ALIGN_PAD_SEC = 0.4          # 第一輪：Whisper 時間前後各多取幾秒
ALIGN_EDGE_SEC = 0.12        # 第一輪結果離搜尋範圍邊緣小於這個值 → 真正的聲音可能在範圍外，進第二輪
ALIGN_WIDE_SEC = 3.0         # 第二輪：往前後各多找幾秒
ALIGN_BATCH_SEC = 20.0       # 第二輪：連續有問題的句子每批最長幾秒
ALIGN_MAX_INNER_GAP = 1.0    # 第二輪結果若同一句相鄰兩字空超過這麼多秒（句子被拉長），不採用
ALIGN_START_LEAD_SEC = 0.0   # 字幕比第一個字提早多少秒出現（想讓字幕早一點點出來可設 0.05～0.1）
ALIGN_BIG_SHIFT_SEC = 1.0    # 對齊後起點移動超過這麼多秒的字幕會列在日誌裡，建議確認
ALIGN_END_HOLD_SEC = 0.25    # 對齊後的結束點剛好落在最後一個字，稍微延長讓字幕不會一閃就消失（不會蓋到下一句）
ALIGN_MIN_DURATION = 0.3     # 單條字幕最短顯示秒數（後面緊接下一條、空隙不夠時就不延長，不會推遲下一條）
ALIGN_LOW_CONF = 0.3         # 對齊信心低於這個值的字幕會列在日誌裡，建議人工檢查
SAMPLE_RATE = 16000

# === 字幕拆分設定（解決 Whisper 把好幾句塞進同一條字幕）===
# 依序套用三種切法：
#   1. 句尾標點（。！？ 等）後面一律切開
#   2. 詞與詞之間停頓 >= SPLIT_PAUSE_SEC 就切開（常常是換人講話或換一句）
#   3. 切完還是太長（字數或秒數超過上限）的，在「最像斷句的地方」再切一刀，
#      優先切在「、」後面，其次是停頓最長、前後長度最平均的地方，直到符合上限
SPLIT_PAUSE_SEC = 0.5        # 停頓超過幾秒就切；覺得切太碎就調大（例如 0.8），切不夠就調小（例如 0.35）
                             # 有對齊時量的是 wav2vec2 的字間空白（字的拉長音也會算進空白），
                             # 沒對齊時量的是 Whisper 的詞間空白，兩者手感略有不同
SPLIT_MAX_CHARS = 28         # 單條字幕最多幾個字（日文字幕一行約 14 字，兩行約 28 字）
SPLIT_MAX_DURATION = 7.0     # 單條字幕最長幾秒
SPLIT_MIN_CHARS = 2          # 依長度切時，兩邊至少要有幾個字（避免切出只有一個字的碎片）
SPLIT_MIN_DISPLAY_SEC = 0.3  # 沒做對齊時，拆出來的字幕最短顯示秒數
SENTENCE_END_CHARS = "。．！？!?♪"
# 句尾標點後面可以緊跟的右括號／引號：切點要放在它們後面，不然「」」會跑到下一條開頭
CLOSING_CHARS = "」』）)】〕］]”’"

# === 開頭空白字幕設定（方便匯入剪輯軟體時對軸）===
# 內容用「零寬空格」：畫面上看不到，但不會像真正的空白一樣被剪輯軟體或 SRT 解析器
# 修剪成空字串、當成空字幕丟掉。如果你的剪輯軟體把它顯示成方框或問號，
# 可以改成全形空格 "　" 或半形句點 "." 試試。
LEADING_BLANK_TEXT = "\u200b"
LEADING_BLANK_MIN_SEC = 0.05   # 第一句字幕離 0 秒不到這麼多時（本來就從開頭開始），就不用插
SOFT_BREAK_CHARS = "、，,…"
# 依長度切時的日文斷句提示：
#   切在這些字尾之後比較自然（子句結束）
CLAUSE_END_SUFFIXES = ("けど", "けれど", "けども", "けれども", "から", "ので", "のに", "たら")
#   切在這些單獨的助詞之後也還算自然（例如「一緒に / 早押しクイズを」）
AFTER_PARTICLES = {"は", "が", "を", "に", "と", "で", "も"}
#   有裝 janome 時，這些單獨假名一律當成助詞（janome 偶爾會把人名後面的「が」判成接續詞）
PARTICLE_SURFACES = {"は", "が", "を", "に", "の", "も", "と", "で", "へ", "や", "よ", "ね", "か", "な", "わ", "さ"}
#   沒裝 janome 時的替代規則：這些詞後面不切（它們是跟後面的話連在一起的）
ATTACH_NEXT_WORDS = ("そして", "それから", "それで", "だから", "でも", "じゃあ", "えーと", "えっと",
                     "あのー", "この", "その", "あの", "まあ")
#   下一條字幕不要用這些單獨的助詞開頭（例如「最近 / はゲーム」這種切法）
NO_START_PARTICLES = {"は", "が", "を", "に", "の", "も", "と", "へ", "や", "よ", "ね", "か", "な", "わ", "さ", "で"}


def _ensure_dir(preferred: str, name: str, label: str) -> str:
    """
    建立 preferred 資料夾並回傳實際可用的路徑。建不起來就依序改用腳本旁邊的 <name>、
    使用者家目錄下的 whisper_subtitles/<name>，並在 STARTUP_NOTES 記下改用了哪裡。
    """
    candidates = [Path(preferred), BASE_DIR / name,
                  Path(os.path.expanduser("~")) / "whisper_subtitles" / name]
    for i, folder in enumerate(candidates):
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"無法建立{label}資料夾 {folder}：{e}")
            continue
        if i > 0:
            STARTUP_NOTES.append(f"注意：{label}資料夾 {preferred} 建立不起來，改用 {folder}")
        return str(folder)
    raise OSError(f"{label}資料夾建立失敗：{preferred}（也試過 {candidates[1]}、{candidates[2]}）")


try:
    output_dir = _ensure_dir(output_dir, "text", "輸出")
    separation_work_dir = _ensure_dir(separation_work_dir, "separated", "人聲分離暫存")
except Exception:
    _show_fatal_error("無法建立輸出資料夾。請檢查 w1_1.py 裡 output_dir／separation_work_dir 的路徑。")
    raise

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)


def format_timestamp(seconds: float) -> str:
    """將秒數轉換為 SRT 標準時間格式 HH:MM:SS,mmm"""
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, milliseconds = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


# 一個詞的時間與文字。Whisper 給的詞、以及對齊後改用 wav2vec2 時間的詞都用這個格式
Word = collections.namedtuple("Word", "start end word")


class JobCancelled(Exception):
    """使用者按了「取消處理」（或關閉視窗），這個檔案不用做了。"""


def _kill_tree(proc) -> None:
    """
    停掉子程序。Windows 上用 taskkill /T 連它底下的子程序一起停（pip 裝的 demucs.exe 只是個啟動器，
    真正在跑的是它開出來的 python.exe）；其他系統直接 kill。已經結束的不做事。
    """
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, creationflags=_NO_WINDOW, timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


class JobControl:
    """
    「取消」的狀態，以及背景子程序（Demucs、對齊）的登記處。

    按一次取消 generation 就加 1：在那之前排進佇列的、正在跑的工作全部作廢。用計數而不是旗標，
    取消剛好發生在工作開始的瞬間也不會漏掉。子程序都要用 spawn() 啟動，取消、關閉視窗、
    程式結束時才能一起停掉，不會留在背景占著 GPU。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._generation = 0
        self._job_generation = 0   # 目前這個工作是在哪個 generation 排進來的
        self._procs = set()

    @property
    def generation(self) -> int:
        return self._generation

    def begin(self, generation: int) -> None:
        """工作開始：記下它是在哪個 generation 排進來的。"""
        self._job_generation = generation

    def is_cancelled(self, generation: int | None = None) -> bool:
        return self._generation != (self._job_generation if generation is None else generation)

    def check(self) -> None:
        """目前的工作被取消了就丟出 JobCancelled。各處理階段之間呼叫。"""
        if self.is_cancelled():
            raise JobCancelled()

    def cancel(self) -> None:
        with self._lock:
            self._generation += 1
            procs = list(self._procs)
        for proc in procs:
            _kill_tree(proc)

    def spawn(self, cmd, **kwargs):
        proc = subprocess.Popen(cmd, **kwargs)
        with self._lock:
            self._procs.add(proc)
        if self.is_cancelled():  # 取消剛好發生在啟動的空檔
            _kill_tree(proc)
        return proc

    def release(self, proc) -> None:
        """子程序用完了：還在跑的就停掉，並從登記處拿掉。"""
        if proc is None:
            return
        _kill_tree(proc)
        with self._lock:
            self._procs.discard(proc)


JOBS = JobControl()
atexit.register(JOBS.cancel)  # 不管怎麼結束，都不要留下背景程序


_PROGRESS_RE = re.compile(r"(\d{1,3})%\|")   # tqdm 進度條的樣子：「 45%|████▌     | …」


def _iter_console_lines(stream):
    """
    逐行讀子程序的輸出（bytes）。tqdm 進度條是用「回到行首」的方式原地更新的，
    所以回車字元也當作換行，不然整條進度會擠成一行、等到結束才讀得到。
    """
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    buf = ""
    while True:
        chunk = stream.read1(4096)
        if not chunk:
            break
        parts = re.split(r"[\r\n]+", buf + decoder.decode(chunk))
        buf = parts.pop()
        for part in parts:
            if part.strip():
                yield part
    buf += decoder.decode(b"", final=True)
    if buf.strip():
        yield buf


def _wav_is_complete(path: Path) -> bool:
    """
    WAV 檔頭記的音訊長度是不是真的都寫進檔案了。輸出到一半被中斷的檔案，
    檔頭記的長度會是 0 或比實際檔案大，這種不能當成已經分離好的結果。
    """
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return False
            pos = 12
            while True:
                chunk_head = f.read(8)
                if len(chunk_head) < 8:
                    return False
                chunk_id, chunk_size = chunk_head[:4], int.from_bytes(chunk_head[4:], "little")
                if chunk_id == b"data":
                    return chunk_size > 0 and pos + 8 + chunk_size <= size
                pos += 8 + chunk_size + (chunk_size & 1)  # 每個區塊都對齊到偶數位元組
                f.seek(pos)
    except OSError:
        return False


def _demucs_command() -> list[str]:
    """
    啟動 Demucs 的指令前半段。優先用「跑這個程式的同一個 Python」的 demucs（python -m demucs.separate），
    這樣用 .vbs 的 PYTHONW_PATH 指定虛擬環境時也找得到；那個環境沒裝 demucs 才退回 PATH 上的 demucs。
    """
    if importlib.util.find_spec("demucs") is not None:
        return [_console_python(), "-m", "demucs.separate"]
    exe = shutil.which("demucs")
    if exe:
        return [exe]
    raise FileNotFoundError("找不到 Demucs。請在執行這個程式的同一個 Python 環境裡執行 pip install demucs"
                            f"（目前用的 Python：{sys.executable}）")


def separate_vocals(input_path: str, work_dir: str, log_func,
                     device: str = "cuda", model_name: str = "htdemucs",
                     segment: int | None = None, progress_func=None) -> str:
    """
    用 Demucs 分離人聲與背景音樂，回傳人聲音軌路徑。分離很耗時，所以同一個檔案分離過就直接重用：
        結果放在 <work_dir>/<模型>/<檔名>__<來源檔大小與修改時間的雜湊>/vocals.wav，
        同名但內容不同的檔案（a.mp4 和 a.mkv、重新剪過的版本）不會共用到彼此的結果。
        Demucs 先輸出到暫存資料夾，確認檔案完整才搬進來，中途被中斷不會留下半個檔案被當成完成。
        舊版放在 <檔名>/vocals.wav 的結果，只要檔案完整、而且比來源檔新，也會沿用。
    progress_func(0～1) 回報分離進度（從 Demucs 輸出的進度條讀取），沒給就不回報。
    """
    input_path = Path(input_path)
    report = progress_func or (lambda frac: None)
    source = input_path.stat()
    key = hashlib.sha1(f"{source.st_size}:{source.st_mtime_ns}".encode()).hexdigest()[:8]
    model_dir = Path(work_dir) / model_name
    final_dir = model_dir / f"{input_path.stem}__{key}"
    legacy_path = model_dir / input_path.stem / "vocals.wav"

    for cached, must_be_newer in ((final_dir / "vocals.wav", False), (legacy_path, True)):
        if (cached.exists() and _wav_is_complete(cached)
                and (not must_be_newer or cached.stat().st_mtime >= source.st_mtime)):
            log_func(f"[人聲分離] 已存在分離結果，跳過分離：{cached}")
            report(1.0)
            return str(cached)

    tmp_out = Path(work_dir) / f"_partial_{key}"
    shutil.rmtree(tmp_out, ignore_errors=True)  # 上次中斷留下的
    cmd = _demucs_command() + [
        "--two-stems", "vocals",
        "-n", model_name,
        "-d", device,
        "-o", str(tmp_out),
        str(input_path),
    ]
    if segment is not None:
        cmd += ["--segment", str(segment)]

    log_func(f"[人聲分離] 開始處理：{input_path.name}（這步驟可能需要幾分鐘，視音訊長度與裝置而定）")
    print(f"[人聲分離] 指令：{' '.join(cmd)}")
    proc = None
    try:
        proc = JOBS.spawn(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=_child_env(), creationflags=_NO_WINDOW,
        )
        tail = collections.deque(maxlen=15)
        last_pct = -1
        for line in _iter_console_lines(proc.stdout):
            m = _PROGRESS_RE.search(line)
            if m:
                pct = min(int(m.group(1)), 100)
                if pct != last_pct:
                    last_pct = pct
                    report(pct / 100)
                continue
            line = line.strip()
            print(line)
            log_func("[人聲分離] " + line)
            tail.append(line)
        returncode = proc.wait()
        JOBS.check()  # 被取消時子程序是被停掉的，回傳碼不是 0，要先判斷這個
        if returncode != 0:
            detail = "\n".join(tail) if tail else "（沒有輸出）"
            raise RuntimeError(f"Demucs 人聲分離失敗（代碼 {returncode}）：\n{detail}")

        produced = next(iter((tmp_out / model_name).glob("*/vocals.wav")), None)
        if produced is None or not _wav_is_complete(produced):
            raise FileNotFoundError(f"Demucs 已結束，但找不到完整的人聲檔：{tmp_out / model_name}")
        shutil.rmtree(final_dir, ignore_errors=True)  # 前一次留下的不完整結果
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(produced.parent, final_dir)
    finally:
        JOBS.release(proc)
        shutil.rmtree(tmp_out, ignore_errors=True)

    report(1.0)
    log_func(f"[人聲分離] 完成，人聲音軌：{final_dir / 'vocals.wav'}")
    return str(final_dir / "vocals.wav")


# === 模型快取：第一次轉錄時載入，之後一直重用 ===
_loaded_model = None


def get_model(log_func) -> WhisperModel:
    global _loaded_model
    if _loaded_model is None:
        log_func(f"載入模型中：{WHISPER_MODEL} …（只有第一次會比較久）")
        _loaded_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE,
                                     compute_type=WHISPER_COMPUTE_TYPE)
        log_func("模型載入完成。")
    return _loaded_model


def build_hotwords(words: list[str]) -> str:
    """把勾選的 hotwords 串成傳給 Whisper 的字串。"""
    if not words:
        return ""
    if HOTWORDS_JA_PUNCTUATION:
        return "、".join(words) + "。"
    return ", ".join(words)


# =========================================================
# === 時間戳對齊：交給獨立程序 align_worker.py 執行 ===
# =========================================================
# PyTorch 和 faster-whisper 各帶一份不同小版本的 cuDNN 9，放在同一個程序裡會出現
# 「Could not load symbol cudnnGetLibConfig. Error code 127」，所以對齊跟 Demucs 一樣
# 用獨立程序跑，主程式完全不 import torch。
ALIGN_WORKER = BASE_DIR / "align_worker.py"


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # Windows：不要為子程序跳出黑色 cmd 視窗


def _child_env(**extra) -> dict:
    """子程序的環境變數：輸出一律用 UTF-8，主程式才不會讀到亂碼。"""
    return dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", **extra)


def _console_python() -> str:
    """用 pythonw 雙擊啟動時，子程序改用同資料夾的 python.exe，stdout 才能正常傳回進度。"""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        candidate = exe.with_name("python.exe")
        if candidate.exists():
            return str(candidate)
    return str(exe)


def run_alignment(audio_path: str, segments: list[dict], log_func, progress_func) -> dict:
    """
    呼叫 align_worker.py 用 wav2vec2 強制對齊。
    segments: [{"start", "end", "text", "words"(可省略，list[Word])}, ...]
    回傳 {"duration", "spans", "confs", "wide", "word_spans"}，每個 list 都跟 segments 一樣長；
    個別片段對不上時該項是 None，不會整份失敗。
    """
    if not ALIGN_WORKER.exists():
        raise FileNotFoundError(f"找不到 {ALIGN_WORKER.name}，請把它放在 {BASE_DIR}")

    log_func(f"[對齊] 讀取音訊：{audio_path}")
    audio = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE

    with tempfile.TemporaryDirectory(prefix="whisper_align_") as tmp:
        tmp = Path(tmp)
        audio_npy = tmp / "audio.npy"
        job_json = tmp / "job.json"
        result_json = tmp / "result.json"

        np.save(audio_npy, audio.astype(np.float32, copy=False))
        del audio
        job_json.write_text(json.dumps({
            "audio_npy": str(audio_npy),
            "segments": [
                {
                    "start": s["start"], "end": s["end"], "text": s["text"],
                    # 有詞的切法就一起送，worker 會傳回每個詞對齊後的時間（給拆分用）
                    "words": [w.word for w in s["words"]] if s.get("words") else None,
                }
                for s in segments
            ],
            "model_name": ALIGN_MODEL_NAME,
            "device": ALIGN_DEVICE,
            "sample_rate": SAMPLE_RATE,
            "params": {
                "pad_sec": ALIGN_PAD_SEC,
                "edge_sec": ALIGN_EDGE_SEC,
                "wide_sec": ALIGN_WIDE_SEC,
                "batch_sec": ALIGN_BATCH_SEC,
                "low_conf": ALIGN_LOW_CONF,
                "max_inner_gap": ALIGN_MAX_INNER_GAP,
            },
        }, ensure_ascii=False), encoding="utf-8")

        env = _child_env(HF_HUB_DISABLE_SYMLINKS_WARNING="1")
        log_func("[對齊] 啟動對齊程序…")
        proc = JOBS.spawn(
            [_console_python(), str(ALIGN_WORKER), str(job_json), str(result_json)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
            creationflags=_NO_WINDOW,
        )
        try:
            tail = collections.deque(maxlen=15)
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                if line.startswith("PROGRESS "):
                    done, total = line.split()[1:3]
                    progress_func(int(done) / max(int(total), 1))
                elif line.startswith("LOG "):
                    log_func("[對齊] " + line[4:])
                elif line.startswith("ERROR "):
                    log_func("!!! [對齊] " + line[6:])
                    tail.append(line[6:])
                elif line.strip():
                    print(line)  # transformers 的警告、下載進度等只寫進 log 檔
                    tail.append(line)
            returncode = proc.wait()
            JOBS.check()  # 被取消時子程序是被停掉的，回傳碼不是 0，要先判斷這個
            if returncode != 0 or not result_json.exists():
                detail = "\n".join(tail) if tail else "（沒有輸出）"
                raise RuntimeError(f"對齊程序失敗（代碼 {returncode}）：\n{detail}")
        finally:
            JOBS.release(proc)

        result = json.loads(result_json.read_text(encoding="utf-8"))

    n = len(segments)
    spans = result["spans"]
    if len(spans) != n:
        raise RuntimeError(f"對齊結果數量不符（送出 {n} 條，收到 {len(spans)} 條）")
    return {
        "duration": duration,
        "spans": spans,
        "confs": result.get("confs") or [None] * n,
        "wide": result.get("wide") or [False] * n,
        # 舊版 align_worker.py 沒有這一項：拆分會退回用 Whisper 的詞時間
        "word_spans": result.get("word_spans") or [None] * n,
    }


def apply_alignment(segments: list[dict], result: dict, log_func):
    """
    把對齊結果寫回 segments（原地修改）：整段的起訖時間，以及每個詞的時間（有 words 的話）。
    對不上的片段保留原本時間。另外記下 orig_start / conf / wide / aligned 給檢查報告用。
    """
    shifts = []
    for seg, span, conf, wide, wspans in zip(segments, result["spans"], result["confs"],
                                             result["wide"], result["word_spans"]):
        seg["orig_start"] = seg["start"]
        seg["conf"] = conf
        seg["wide"] = bool(wide)
        seg["aligned"] = span is not None
        if span is None:
            continue
        shifts.append(abs(span[0] - seg["start"]))
        orig_start, orig_end = seg["start"], seg["end"]
        seg["start"], seg["end"] = span[0], span[1]
        words = seg.get("words")
        if words and wspans and len(wspans) == len(words):
            new_words, prev_end = [], span[0]
            for w, t in zip(words, wspans):
                if t is None:  # 純標點之類不會發音的詞：時間黏在前一個詞後面
                    new_words.append(Word(prev_end, prev_end, w.word))
                else:
                    new_words.append(Word(t[0], t[1], w.word))
                    prev_end = t[1]
            seg["words"] = new_words
        elif words:
            # 沒拿到詞時間（例如還在用舊版 align_worker.py）：把 Whisper 的詞時間
            # 等比例搬進對齊後的範圍，至少整段的起訖會是對齊後的時間，不會被詞時間蓋回去
            scale = (span[1] - span[0]) / (orig_end - orig_start) if orig_end > orig_start else 0.0

            def remap(t: float) -> float:
                return min(max(span[0] + (t - orig_start) * scale, span[0]), span[1])

            seg["words"] = [Word(remap(w.start), remap(w.end), w.word) for w in words]

    avg_shift = sum(shifts) / len(shifts) if shifts else 0.0
    log_func(f"[對齊] 完成：{len(shifts)}/{len(segments)} 條成功對齊，起點平均移動 {avg_shift:.2f} 秒")


def report_alignment(segments: list[dict], subs: list[dict], log_func):
    """
    列出需要人工確認的片段：對不上的、信心偏低的（通常是 Whisper 聽錯字或幻覺）、
    起點被大幅移動的（多半是修正，但值得確認）。編號是輸出 SRT 裡的字幕編號，
    一段被拆成好幾條時顯示成「#12～14」。subs 要傳「最後真正寫進 SRT 的列表」
    （包含開頭空白字幕），編號才會跟 SRT 對得上；沒有 src 的字幕（空白字幕）不列。
    """
    numbers: dict[int, list[int]] = {}
    for no, sub in enumerate(subs, start=1):
        if sub.get("src") is not None:
            numbers.setdefault(sub["src"], []).append(no)

    suspicious = []
    for i, seg in enumerate(segments):
        if not seg["text"].strip():
            continue
        conf = seg.get("conf")
        if not seg.get("aligned"):
            why = "對不上，保留原時間"
        elif conf is not None and conf < ALIGN_LOW_CONF:
            why = f"信心 {conf:.2f}"
        elif abs(seg["start"] - seg["orig_start"]) >= ALIGN_BIG_SHIFT_SEC:
            why = (f"起點移動 {seg['start'] - seg['orig_start']:+.1f} 秒"
                   + ("，大範圍重對" if seg.get("wide") else ""))
        else:
            continue
        nos = numbers.get(i, [i + 1])
        label = f"#{nos[0]}" if len(nos) == 1 else f"#{nos[0]}～{nos[-1]}"
        suspicious.append(f"    {label} [{format_timestamp(seg['start'])}]（{why}）{seg['text']}")

    if suspicious:
        log_func(f"[對齊] 以下 {len(suspicious)} 處建議檢查（時間可能不準，或字幕文字跟實際說的不一樣）：")
        for line in suspicious[:30]:
            log_func(line)
        if len(suspicious) > 30:
            log_func(f"    …還有 {len(suspicious) - 30} 處，請看 whisper_app.log")
            for line in suspicious[30:]:
                print(line)


def finalize_aligned_timing(subs: list[dict], duration: float):
    """
    對齊後的後處理：稍微提早出現、不重疊、最短顯示時間、結尾稍微延長。
    最短顯示時間與結尾延長都只用到跟下一句之間的空隙：不會蓋到下一句，
    也不會把下一句的起點往後推（空隙不夠就少延長，寧可短一點）。
    先把所有起點提早，再處理重疊與延長，後一句提早的空間才不會被前一句的延長吃掉。
    """
    for sub in subs:
        if sub.get("aligned"):
            sub["start"] = max(0.0, sub["start"] - ALIGN_START_LEAD_SEC)
    for i, sub in enumerate(subs):
        if i > 0 and sub["start"] < subs[i - 1]["end"]:
            sub["start"] = subs[i - 1]["end"]
        next_start = subs[i + 1]["start"] if i + 1 < len(subs) else duration
        want = max(sub["end"], sub["start"] + ALIGN_MIN_DURATION) + ALIGN_END_HOLD_SEC
        sub["end"] = max(sub["end"], min(want, next_start))
        if sub["end"] <= sub["start"]:
            # 前一句蓋過了整句的極端情況：至少給最短顯示時間，後一句的起點下一輪會被推開
            sub["end"] = sub["start"] + ALIGN_MIN_DURATION


# =========================================================
# === SRT 讀寫 ===
# =========================================================
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
    if first_start < LEADING_BLANK_MIN_SEC:
        log_func("第一句字幕從影片開頭就開始了，不需要插入空白字幕")
        return
    subs.insert(0, {"start": 0.0, "end": first_start, "text": LEADING_BLANK_TEXT})
    log_func(f"已在最前面插入空白字幕（00:00:00,000 --> {format_timestamp(first_start)}）")


def write_srt(path, segments: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            f.write(
                f"{idx}\n{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n"
                f"{seg['text']}\n\n"
            )


def align_existing_srt(srt_path: str, media_path: str, use_sep: bool,
                       log_func, progress_func, add_blank: bool = True) -> str:
    """只對齊：讀取現有 SRT 的文字與大略時間，用影片音訊重新校正時間戳，不跑 Whisper。"""
    srt_path = Path(srt_path)
    media_path = Path(media_path)

    log_func(f"\n=== 只對齊時間戳：{srt_path.name} ===")
    log_func(f"對應影片／音訊：{media_path}")

    segments = parse_srt(srt_path)
    # 之前加的開頭空白字幕（或其他看不到字的字幕）沒東西可對齊，先拿掉；需要的話最後再重新加
    n_blank = sum(1 for seg in segments if is_blank_text(seg["text"]))
    segments = [seg for seg in segments if not is_blank_text(seg["text"])]
    if not segments:
        raise ValueError("SRT 裡沒有讀到任何字幕（格式可能不正確）")
    log_func(f"讀到 {len(segments)} 條字幕"
             + (f"（另有 {n_blank} 條空白字幕，已略過）" if n_blank else ""))

    # 有做人聲分離時，分離佔進度條的前一段，對齊縮進剩下的部分
    overall_progress = progress_func
    if use_sep:
        audio_input = separate_vocals(
            str(media_path), separation_work_dir, log_func,
            device=SEPARATION_DEVICE, model_name=SEPARATION_MODEL,
            segment=SEPARATION_SEGMENT,
            progress_func=lambda frac: overall_progress(SEPARATION_PROGRESS_SHARE * frac),
        )
        progress_func = lambda frac: overall_progress(
            SEPARATION_PROGRESS_SHARE + (1 - SEPARATION_PROGRESS_SHARE) * frac)
    else:
        audio_input = str(media_path)
    JOBS.check()

    result = run_alignment(audio_input, segments, log_func, progress_func)
    apply_alignment(segments, result, log_func)
    # 輸出用另一份，檢查報告才看得到後處理（提早出現、防重疊）之前的對齊時間
    subs = [{"start": seg["start"], "end": seg["end"], "text": seg["text"],
             "src": i, "aligned": seg["aligned"]} for i, seg in enumerate(segments)]
    finalize_aligned_timing(subs, result["duration"])
    if add_blank:
        add_leading_blank(subs, log_func)
    report_alignment(segments, subs, log_func)

    JOBS.check()
    output_path = os.path.join(output_dir, f"{srt_path.stem}_align.srt")
    write_srt(output_path, subs)
    progress_func(1.0)
    log_func(f"=== 完成，對齊後字幕已儲存：{output_path} ===\n")
    return output_path


# =========================================================
# === 字幕拆分：用 word timestamps 把塞了好幾句的片段拆開 ===
# =========================================================
# ---- 哪些「詞與詞之間」可以切 ----
# Whisper 給的日文「詞」其實是 token：「熊本」會變成「熊」「本」兩個。對齊後的停頓又是用
# wav2vec2 的字間空白量的，一個漢字念好幾拍（熊＝くま），同一個詞的兩個漢字之間也可能
# 看起來像停頓。所以先用 janome 斷詞，只允許在「文節」的交界切：
#   ・不在一個詞的中間切（熊｜本）
#   ・助詞、助動詞、接尾詞、標點黏在前面（熊本｜における、8｜年）
#   ・接續詞、連體詞、接頭詞、フィラー黏在後面（そして｜今回の、その｜問題）
#   ・複合名詞不拆（熊本｜地震、早押し｜クイズ），「名詞＋する」不拆（勉強｜する）
try:
    from janome.tokenizer import Tokenizer as _JanomeTokenizer
except ImportError:
    _JanomeTokenizer = None
_janome = None


def _get_janome():
    """第一次用到才建立。hotwords 候選詞加進使用者字典，人名和「早押し」這類詞才不會被拆錯。"""
    global _janome
    if _janome is None and _JanomeTokenizer is not None:
        with tempfile.TemporaryDirectory(prefix="whisper_janome_") as tmp:
            udic = os.path.join(tmp, "hotwords.csv")
            with open(udic, "w", encoding="utf-8") as f:
                for w in HOTWORD_NAME_CANDIDATES + HOTWORD_COMMON_CANDIDATES:
                    if "," not in w:
                        f.write(f"{w},名詞,{w}\n")
            _janome = _JanomeTokenizer(udic, udic_type="simpledic", udic_enc="utf8")
    return _janome


def split_boundary_mode() -> str:
    return "janome 斷詞" if _JanomeTokenizer is not None else "簡易規則（建議 pip install janome）"


def _can_break(prev: list[str], cur: list[str]) -> bool:
    """兩個 janome 詞（品詞欄位 list）之間是不是文節交界。"""
    if cur[0] in ("助詞", "助動詞"):
        return False
    if cur[0] == "記号" and cur[1] != "括弧開":
        return False
    if cur[1] in ("接尾", "非自立"):
        return False
    if prev[0] in ("接頭詞", "連体詞", "接続詞", "フィラー"):
        return False
    if prev[0] == "記号" and prev[1] == "括弧開":
        return False
    if prev[0] == "名詞" and cur[0] == "名詞" and prev[1] != "副詞可能":
        return False  # 複合名詞（「今日」「今回」這類時間名詞後面可以切）
    if prev[0] == "名詞" and prev[1] == "サ変接続" and cur[0] == "動詞":
        return False
    return True


def _janome_break_offsets(text: str) -> set[int]:
    """回傳 text 裡可以切開的字元位置。"""
    spans = []  # 不含空白的詞：(起點, 終點, 品詞欄位)
    pos = 0
    for t in _get_janome().tokenize(text):
        start, pos = pos, pos + len(t.surface)
        fields = t.part_of_speech.split(",")
        if fields[0] == "記号" and fields[1] == "空白":
            continue
        if t.surface in PARTICLE_SURFACES:
            fields = ["助詞", "*"]
        spans.append((start, pos, fields))
    ok = set()
    for (_, e1, p1), (s2, _, p2) in zip(spans, spans[1:]):
        if _can_break(p1, p2):
            ok.update(range(e1, s2 + 1))  # 中間有空白的話，空白前後都算
    return ok


def _char_class(ch: str) -> str:
    if ch == "々" or "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
        return "kanji"
    if "\u30a1" <= ch <= "\u30ff" or "\uff66" <= ch <= "\uff9f":
        return "katakana"
    if unicodedata.normalize("NFKC", ch).isascii() and ch.isalnum():
        return "alnum"
    return "other"


def _heuristic_can_break(left_text: str, right_word: str) -> bool:
    """沒裝 janome 時的替代規則：只看切點兩邊的字。"""
    a, b = left_text.rstrip(), right_word.lstrip()
    if not a or not b:
        return True
    if b[0] in "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶーｰ゛゜" or b[0] in CLOSING_CHARS or b in PARTICLE_SURFACES:
        return False
    ca, cb = _char_class(a[-1]), _char_class(b[0])
    if ca == cb and ca in ("kanji", "katakana", "alnum") and not right_word[:1].isspace():
        return False  # 熊｜本、ツア｜ー、Quiz｜Knock
    if ca == "alnum" and cb == "kanji":
        return False  # 8｜年
    return not a.endswith(ATTACH_NEXT_WORDS)


def _allowed_cuts(words) -> list[bool]:
    """每一對相鄰的詞（words[k] 和 words[k+1]）之間能不能切，長度 = len(words) - 1。"""
    if len(words) < 2:
        return []
    if _JanomeTokenizer is not None:
        ok = _janome_break_offsets("".join(w.word for w in words))
        result, end = [], 0
        for w in words[:-1]:
            end += len(w.word)
            result.append(end in ok)
        return result
    result, left = [], ""
    for w, nxt in zip(words, words[1:]):
        left += w.word
        result.append(_heuristic_can_break(left, nxt.word))
    return result


def _char_count(s: str) -> int:
    """字數（不算空白）。"""
    return len(re.sub(r"\s", "", s))


def _words_text(words) -> str:
    return "".join(w.word for w in words).strip()


def _last_char(word) -> str:
    t = word.word.strip()
    return t[-1] if t else ""


def _sentence_end_flags(words) -> list[bool]:
    """
    每個詞是不是「句子在這裡結束」。句尾標點後面緊跟右括號／引號時，句尾記在最後一個
    括號上（。」→ 切在」之後）；標點跟括號黏在同一個詞裡（。」）也認得。
    """
    flags = []
    for w in words:
        t = w.word.strip()
        core = t.rstrip(CLOSING_CHARS)
        if core:
            flags.append(core[-1] in SENTENCE_END_CHARS)
        elif t and flags:  # 這個詞只有括號：沿用前一個詞的句尾狀態，切點往後移
            flags[-1], moved = False, flags[-1]
            flags.append(moved)
        else:
            flags.append(False)
    return flags


def _too_long(words) -> bool:
    return (_char_count(_words_text(words)) > SPLIT_MAX_CHARS
            or words[-1].end - words[0].start > SPLIT_MAX_DURATION)


def _best_cut(words, ok: list[bool]) -> int | None:
    """
    在過長的片段裡找最適合切開的位置，回傳「切在第 k 個詞之後」的 k。
    分數 = 停頓秒數 + （前一個詞以「、」或けど／から等結尾就加分）
          − （下一條會以單獨助詞開頭就扣分）+ （前後長度越平均越加分）。
    兩邊都至少要有 SPLIT_MIN_CHARS 個字，而且只考慮 ok[k] 為 True（文節交界）的位置；
    整段都沒有合法位置（例如一長串複合名詞）才退回任意位置。找不到回傳 None。
    """
    for restrict in (True, False):
        k = _best_cut_pass(words, ok if restrict else None)
        if k is not None:
            return k
    return None


def _best_cut_pass(words, ok: list[bool] | None) -> int | None:
    total = _char_count(_words_text(words))
    best_k, best_score = None, float("-inf")
    left = 0
    for k in range(len(words) - 1):
        left += _char_count(words[k].word)
        right = total - left
        if left < SPLIT_MIN_CHARS or right < SPLIT_MIN_CHARS:
            continue
        if ok is not None and not ok[k]:
            continue
        score = max(0.0, words[k + 1].start - words[k].end)
        if _last_char(words[k]) in SOFT_BREAK_CHARS:
            score += 1.0
        elif words[k].word.strip().endswith(CLAUSE_END_SUFFIXES):
            score += 0.6
        elif words[k].word.strip() in AFTER_PARTICLES:
            score += 0.3
        if words[k + 1].word.strip() in NO_START_PARTICLES:
            score -= 1.0
        score += 0.5 * (1 - abs(left - right) / max(total, 1))
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _split_long(words, ok: list[bool]) -> list:
    """太長就切一刀，兩半各自再檢查，直到都符合上限（或已經切不動）。ok 長度 = len(words) - 1。"""
    if len(words) < 2 or not _too_long(words):
        return [words]
    k = _best_cut(words, ok)
    if k is None:
        return [words]
    return _split_long(words[:k + 1], ok[:k]) + _split_long(words[k + 1:], ok[k + 1:])


def split_segment(seg: dict) -> list[dict]:
    """
    把一個片段 {"start", "end", "text", "words": list[Word] 或 None} 拆成多條字幕，
    回傳 [{"start", "end", "text"}, ...]。沒有詞時間時原樣回傳一條。
    words 的時間可能是 Whisper 估的，也可能是對齊後 wav2vec2 的（apply_alignment 換過）。
    """
    words = [w for w in (seg.get("words") or []) if w.word.strip()]
    if not words:
        return [{"start": seg["start"], "end": seg["end"], "text": seg["text"]}]

    ok = _allowed_cuts(words)  # 哪些位置是文節交界（不會切在一個詞的中間）
    ends = _sentence_end_flags(words)  # 哪些詞是句尾（已把後面的右括號算進去）

    # 第 1、2 步：句尾標點 → 切；明顯停頓而且剛好在文節交界 → 切
    groups, start = [], 0
    for i in range(len(words) - 1):
        w, nxt = words[i], words[i + 1]
        if ends[i]:
            cut = True
        else:
            cut = (ok[i] and nxt.start - w.end >= SPLIT_PAUSE_SEC
                   and _char_count(_words_text(words[start:i + 1])) >= SPLIT_MIN_CHARS)
        if cut:
            groups.append((start, i + 1))
            start = i + 1
    groups.append((start, len(words)))

    # 第 3 步：還是太長的再切
    pieces = []
    for a, b in groups:
        for part in _split_long(words[a:b], ok[a:b - 1]):
            pieces.append({"start": part[0].start, "end": part[-1].end, "text": _words_text(part)})
    return pieces


def _ensure_min_display(segments: list[dict]):
    """沒做對齊時，避免拆出來的字幕長度是 0 或一閃即逝（不會蓋到下一條）。"""
    for i, seg in enumerate(segments):
        if seg["end"] - seg["start"] >= SPLIT_MIN_DISPLAY_SEC:
            continue
        limit = segments[i + 1]["start"] if i + 1 < len(segments) else float("inf")
        seg["end"] = max(seg["end"], min(seg["start"] + SPLIT_MIN_DISPLAY_SEC, limit))


def process_file(file_path: str, use_sep: bool, use_align: bool, use_split: bool,
                  hotword_list: list[str],
                  log_func, progress_func, add_blank: bool = True) -> str:
    """處理單一檔案：（可選）人聲分離 -> 轉錄 ->（可選）時間戳對齊 -> 輸出 SRT，回傳輸出檔案路徑。"""
    input_path = Path(file_path)
    base = input_path.stem  # 用原始檔名作為輸出檔名的基礎，取代固定的 "abc"

    sep_tag = "sep" if use_sep else "nosep"

    log_func(f"\n=== 開始處理：{input_path.name} ===")
    log_func(
        f"設定：模型 = {WHISPER_MODEL}、人聲分離 = {'是' if use_sep else '否'}、"
        f"時間戳對齊 = {'是' if use_align else '否'}、自動拆分 = {'是' if use_split else '否'}"
    )

    # 有做人聲分離時，分離佔進度條的前一段，轉錄與對齊縮進剩下的部分
    overall_progress = progress_func
    if use_sep:
        transcribe_input = separate_vocals(
            str(input_path), separation_work_dir, log_func,
            device=SEPARATION_DEVICE, model_name=SEPARATION_MODEL,
            segment=SEPARATION_SEGMENT,
            progress_func=lambda frac: overall_progress(SEPARATION_PROGRESS_SHARE * frac),
        )
        progress_func = lambda frac: overall_progress(
            SEPARATION_PROGRESS_SHARE + (1 - SEPARATION_PROGRESS_SHARE) * frac)
    else:
        transcribe_input = str(input_path)
    JOBS.check()

    model = get_model(log_func)

    # hotwords 由前端勾選傳入；每部影片出場的人不同，只放這次用得到的詞
    hotwords = build_hotwords(hotword_list)
    log_func(f"本次使用的 hotwords：{hotwords if hotwords else '（無）'}")

    segments, info = model.transcribe(
        transcribe_input,
        **TRANSCRIBE_OPTIONS,
        word_timestamps=use_split,  # 拆分需要每個詞的時間
        hotwords=hotwords or None,
    )

    log_func(f"轉錄來源: {transcribe_input}")
    log_func(f"音訊總長度: {info.duration:.2f} 秒")
    log_func(f"VAD 後語音長度: {info.duration_after_vad:.2f} 秒")

    # 要做對齊時，轉錄佔進度條前 80%，對齊佔後 20%
    transcribe_weight = 0.8 if use_align else 1.0
    total = max(info.duration, 0.01)
    raw = []  # Whisper 原本的片段（還沒拆）
    for segment in segments:
        JOBS.check()  # Whisper 是在這個迴圈裡一段一段轉錄的，取消最慢等到目前這一段做完
        text = segment.text.strip()
        words = ([Word(w.start, w.end, w.word) for w in segment.words]
                 if use_split and segment.words else None)
        raw.append({"start": segment.start, "end": segment.end, "text": text, "words": words})
        log_func(f"[{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}] {text}")
        progress_func(min(segment.end / total, 1.0) * transcribe_weight)

    # 先對齊 Whisper 原本的整段（文字長對得穩），順便拿到每個詞的精確時間
    align_result = None
    if use_align and raw:
        log_func("[對齊] 開始校正時間戳…")
        try:
            align_result = run_alignment(
                transcribe_input, raw, log_func,
                lambda frac: progress_func(transcribe_weight + (1 - transcribe_weight) * frac),
            )
        except JobCancelled:
            raise
        except Exception as e:
            traceback.print_exc()
            log_func(f"!!! [對齊] 失敗：{e}（這次先輸出未對齊的字幕）")
        if align_result is not None:
            apply_alignment(raw, align_result, log_func)

    # 再拆：有對齊的片段用 wav2vec2 的詞時間判斷停頓，沒對齊的用 Whisper 的詞時間
    subs = []
    for i, seg in enumerate(raw):
        if use_split:
            pieces = split_segment(seg)
        else:
            pieces = [{"start": seg["start"], "end": seg["end"], "text": seg["text"]}]
        for piece in pieces:
            piece["src"] = i
            piece["aligned"] = seg.get("aligned", False)
        if len(pieces) > 1:
            log_func(f"[拆分] {seg['text']}\n    → " + " ／ ".join(p["text"] for p in pieces))
        subs.extend(pieces)
    if use_split:
        log_func(f"[拆分] Whisper 原本 {len(raw)} 段 → 拆成 {len(subs)} 條字幕"
                 f"（切點判斷：{split_boundary_mode()}）")

    if align_result is not None:
        finalize_aligned_timing(subs, align_result["duration"])
    elif use_split:
        _ensure_min_display(subs)
    if add_blank:
        add_leading_blank(subs, log_func)
    if align_result is not None:
        report_alignment(raw, subs, log_func)  # 插完空白字幕再列，編號才跟 SRT 一致

    align_tag = "_align" if align_result is not None else ""
    JOBS.check()
    output_filename = f"{base}_{WHISPER_MODEL}_{sep_tag}{align_tag}.srt"
    output_path = os.path.join(output_dir, output_filename)
    write_srt(output_path, subs)

    progress_func(1.0)
    log_func(f"=== 完成，字幕已儲存：{output_path} ===\n")
    return output_path


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
                            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in MEDIA_EXTENSIONS),
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
        elif suffix in MEDIA_EXTENSIONS:
            add(media, p)
        else:
            skipped.append(p.name)
    return media, srts, skipped


# =========================================================
# === GUI 佈景主題（QuizKnock 風格：白／紅／黑）===
# =========================================================
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


# =========================================================
# === GUI：會跟著視窗寬度自動排版的勾選框格子 ===
# =========================================================
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


# =========================================================
# === GUI：空間不夠時才出現捲軸的容器 ===
# =========================================================
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


# =========================================================
# === GUI：拖放視窗 ===
# =========================================================
class App:
    # 視窗大小以 100% 縮放（96 DPI）設計；螢幕不夠大時會自動縮小，保證整個視窗都在螢幕內
    WIN_W, WIN_H = 1120, 680
    MIN_W, MIN_H = 860, 560
    LEFT_W = 520        # 左欄（設定／Hotwords／拖放區）寬度，視窗拉寬時多出來的空間都給右邊日誌
    SCREEN_RESERVE = 110  # 螢幕高度要預留給工作列與視窗標題列的空間

    def __init__(self, root):
        self.root = root
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
        for note in STARTUP_NOTES:
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

        self.use_sep_var = tk.BooleanVar(value=True)
        self.use_align_var = tk.BooleanVar(value=True)
        self.use_split_var = tk.BooleanVar(value=True)
        self.add_blank_var = tk.BooleanVar(value=True)
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
        self._hotword_row(frame, "人名", HOTWORD_NAME_CANDIDATES, pady=(6, 2))
        self._hotword_row(frame, "通用詞", HOTWORD_COMMON_CANDIDATES, pady=(6, 8))

    def _hotword_row(self, parent, tag: str, words: list[str], pady):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", padx=10, pady=pady)
        # 兩個分類標籤設成相同最小寬度，右邊的勾選框起點才會對齊
        ttk.Label(
            row, text=tag, style="Tag.TLabel", width=-6, anchor="center",
        ).pack(side="left", anchor="n", pady=(2, 0))
        grid = FlowGrid(row, style="Card.TFrame")
        grid.pack(side="left", fill="x", expand=True, padx=(8, 0))
        for word in words:
            var = tk.BooleanVar(value=True)
            self.hotword_vars[word] = var
            grid.add(ttk.Checkbutton(grid, text=word, variable=var, style="Card.TCheckbutton"))

    # ---- 左欄：拖放區（填滿左欄剩下的高度，視窗拉高時目標也跟著變大）----
    def _build_drop_zone(self, parent):
        self.drop_zone = tk.Frame(
            parent, bg=Theme.WHITE, height=self._px(110),
            highlightthickness=3, highlightbackground=Theme.RED, highlightcolor=Theme.RED,
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
            bordercolor=Theme.BLACK, borderwidth=2, relief="solid",
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
            font=(Theme.FONT_FAMILY, 9, "bold"), padding=(6, 2),
        )

        style.configure(
            "TCheckbutton", background=Theme.WHITE, foreground=Theme.BLACK, font=Theme.BODY_FONT,
        )
        style.configure(
            "Card.TCheckbutton", background=Theme.WHITE, foreground=Theme.BLACK, font=Theme.BODY_FONT,
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
            padding=(14, 6),
        )
        style.map(
            "Red.TButton",
            background=[("active", Theme.RED_DARK), ("pressed", Theme.RED_DARK)],
            foreground=[("active", Theme.WHITE)],
        )

        # 次要按鈕：白底紅字紅框，跟主按鈕做出層級區分
        style.configure(
            "RedOutline.TButton", background=Theme.WHITE, foreground=Theme.RED,
            font=Theme.BUTTON_FONT, borderwidth=1.5, bordercolor=Theme.RED,
            padding=(14, 6),
        )
        style.map(
            "RedOutline.TButton",
            background=[("active", Theme.RED_SOFT), ("pressed", Theme.RED_SOFT)],
            foreground=[("active", Theme.RED)],
        )

        # 小尺寸按鈕（放在外框標題列）：「Small.Red.TButton」會沿用 Red.TButton 的顏色，只改大小
        style.configure("Small.Red.TButton", font=Theme.SMALL_BUTTON_FONT, padding=(10, 1))
        style.configure("Small.RedOutline.TButton", font=Theme.SMALL_BUTTON_FONT, padding=(10, 1))

        # 進度條：紅色前景、淡灰底、黑色外框
        style.configure(
            "Red.Horizontal.TProgressbar",
            troughcolor=Theme.OFF_WHITE, background=Theme.RED,
            bordercolor=Theme.BLACK, lightcolor=Theme.RED, darkcolor=Theme.RED,
            thickness=14,
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
            text_col, text=f"拖放檔案即可自動轉錄・Whisper {WHISPER_MODEL}",
            bg=Theme.RED, fg=Theme.WHITE, font=Theme.SUBTITLE_FONT, anchor="w",
        ).pack(fill="x")

        tk.Frame(self.root, bg=Theme.BLACK, height=3).pack(fill="x")

    # ---- 給背景執行緒呼叫，把更新丟回主執行緒（不可直接從子執行緒操作 Tk 元件）----
    def log(self, msg: str):
        self.ui_queue.put(("log", msg))

    def set_progress(self, frac: float):
        self.ui_queue.put(("progress", frac))

    def set_status(self, msg: str):
        self.ui_queue.put(("status", msg))

    def poll_ui_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "log":
                    self.log_box.configure(state="normal")
                    self.log_box.insert("end", str(payload) + "\n")
                    self.log_box.see("end")
                    self.log_box.configure(state="disabled")
                elif kind == "progress":
                    self.progress["value"] = float(payload) * 100
                elif kind == "status":
                    self.status_var.set(" " + str(payload))
        except queue.Empty:
            pass
        finally:
            # 就算處理某一筆時出錯，也要讓計時器繼續跑，不然畫面之後就不會更新了
            self.root.after(100, self.poll_ui_queue)

    def _set_all_hotwords(self, checked: bool):
        for var in self.hotword_vars.values():
            var.set(checked)

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
                            ("影片／音訊", " ".join("*" + ext for ext in sorted(MEDIA_EXTENSIONS))),
                            ("所有檔案", "*.*"),
                        ],
                    )
                if not media:
                    self.log(f">>> 略過「{Path(srt).name}」：沒有選擇對應的影片")
                    continue
                used_media.add(media)
                self._enqueue({"kind": "align_only", "srt": srt, "media": media,
                               "use_sep": use_sep, "add_blank": add_blank})
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
        if len(media_files) > MANY_FILES_CONFIRM and not messagebox.askyesno(
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
                    align_existing_srt(job["srt"], job["media"], job["use_sep"],
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
                where = f"（詳細內容在 {LOG_FILE}）" if LOG_FILE else ""
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
        JOBS.cancel()
        self.root.destroy()

    def _report_callback_exception(self, exc, val, tb):
        """Tk 事件處理函式（拖放、按鈕、計時器）裡沒接住的例外：寫進 log 檔，並提示使用者。"""
        traceback.print_exception(exc, val, tb)
        key = f"{exc.__name__}: {val}"
        if key in self._reported_errors:  # 同一個錯誤（例如計時器每 0.1 秒一次）只提示一次
            return
        self._reported_errors.add(key)
        where = f"\n\n詳細錯誤已寫入：\n{LOG_FILE}" if LOG_FILE else ""
        self.log(f"!!! 介面發生錯誤：{key}")
        try:
            messagebox.showerror("Whisper 字幕產生器 - 發生錯誤", f"{key}{where}", parent=self.root)
        except Exception:
            pass


def main():
    root = TkinterDnD.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _show_fatal_error("應用程式啟動或執行時發生未預期的錯誤。")