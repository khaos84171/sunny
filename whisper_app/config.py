"""
所有「可以調整的設定」都在這個檔案。改完存檔，重新開啟程式就生效。
（設定的說明寫在每一項旁邊；視窗上的勾選狀態不在這裡，會自動記在 whisper_settings.json。）
"""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent   # 放 w1_1.py 的那個資料夾

# log 檔超過這個大小就換檔（whisper_app.log → .log.1 → .log.2 …），只留最近幾份，不會無限長大
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUPS = 3
ALLOW_MULTIPLE_INSTANCES = False   # False = 同時只能開一個視窗（兩個會同時寫同一個 log 與設定檔、各載入一份模型搶顯存）
SEPARATION_DEVICE = "auto"        # "auto" = 有 NVIDIA GPU 就用，沒有就用 CPU（也可寫死 "cuda"／"cpu"）
                                   # 用 GPU 時是與 Whisper 共用，注意顯存是否足夠
SEPARATION_MODEL = "htdemucs"     # htdemucs_ft 是 4 個模型的組合，記憶體需求約 4 倍，容易爆記憶體
                                   # 若記憶體充足想換回更高品質版本，可改成 "htdemucs_ft"
SEPARATION_SEGMENT = 7            # htdemucs 是 Transformer 架構，硬性上限 7.8 秒，不能設更大
                                   # 只能往下調（例如 5）來進一步省記憶體；設 None 則不加此參數
SEPARATION_PROGRESS_SHARE = 0.25  # 有勾人聲分離時，分離佔進度條的前多少比例（轉錄、對齊縮進剩下的部分）

# 偏好的資料夾。建立不起來（例如這台電腦沒有 D 槽）會自動改用腳本旁邊的 text／separated，
# 實際用到的位置會寫進視窗開啟後日誌的第一行。
OUTPUT_DIR = r"D:\桌面\whisper\text"
SEPARATION_WORK_DIR = r"D:\桌面\whisper\separated"

# 拖放進來的影片／音訊只接受這些副檔名（拖資料夾進來時，只會挑出裡面符合的）；沒列到的格式可自己加進來
MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".ts", ".mts", ".m2ts", ".flv", ".wmv", ".m4v",
    ".mpg", ".mpeg", ".3gp", ".ogv",
    ".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif",
}
MANY_FILES_CONFIRM = 30   # 一次要加入超過這麼多個檔案（例如不小心拖了整個資料夾）時，先問一下
DEBUG_LOG = False         # True = faster_whisper 的詳細除錯訊息（每個語音片段、每次解碼）也寫進 log 檔；檔案會長得很快
LOG_MAX_LINES = 3000      # 視窗右側「進度與日誌」最多留幾行，超過就從最舊的開始刪（完整內容還在 log 檔裡）
SETTINGS_FILE = BASE_DIR / "whisper_settings.json"   # 記住視窗上的勾選狀態與 Hotwords

# === Whisper 設定（固定使用 large-v3）===
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "auto"            # "auto" = 有 NVIDIA GPU 就用，沒有就改用 CPU（也可寫死 "cuda"／"cpu"）
WHISPER_COMPUTE_TYPE = "float16"   # 用 GPU 時的計算精度
WHISPER_CPU_COMPUTE_TYPE = "int8"  # 用 CPU 時的計算精度（CPU 不支援 float16）；large-v3 跑在 CPU 上會慢很多

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
    # 只有在有詞時間時（勾了「自動拆分」）才會生效：偵測到可能是幻覺時，跳過超過這麼多秒的靜音再繼續。
    # 不需要就設成 None
    hallucination_silence_threshold=2.0,
)

# Whisper 在靜音或背景音樂上常常編出固定的片語（例如影片結尾的「ご視聴ありがとうございました」）。
# 整條字幕「只有」下面這些片語之一，而且 Whisper 自己也覺得那裡不像有人說話（無語音機率高，或平均信心低）
# 才會略過；真的有人說這句話時無語音機率會很低，不會被誤刪。被略過的每一筆都會寫在日誌。
FILTER_HALLUCINATIONS = True
HALLUCINATION_MIN_NO_SPEECH_PROB = 0.4   # 無語音機率達到這個值…
HALLUCINATION_MAX_AVG_LOGPROB = -1.0     # …或平均 log 機率低於這個值，才當成幻覺
HALLUCINATION_PHRASES = [
    "ご視聴ありがとうございました", "ご視聴ありがとうございます", "最後までご視聴いただきありがとうございました",
    "ご覧いただきありがとうございました", "チャンネル登録お願いします", "チャンネル登録をお願いします",
    "チャンネル登録よろしくお願いします", "高評価お願いします",
]

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
ALIGN_DEVICE = "auto"        # "auto" = 有 NVIDIA GPU 就用，沒有就用 CPU（也可寫死 "cuda"／"cpu"）
ALIGN_KEEP_ALIVE_SEC = 120   # 對齊完後對齊程序留著等下一個檔案幾秒（省掉重新載入模型）；0 = 每個檔案都重新載入

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

# === 用聲音能量微調邊界 + 交叉驗證 ===
# CTC 只會在「模型確定聽到那個字」的一兩格（每格 20 毫秒）標出位置，通常落在母音附近：
# 詞的起點常常比真正開口晚 0.03～0.1 秒，終點比聲音真正結束早 0.05～0.15 秒。
# 這裡在 CTC 的結果附近，用聲音能量（獨立於文字與模型的量測）找「靜音 → 有聲」和「有聲 → 靜音」的位置，
# 把每個詞的起訖貼過去；同時當作交叉驗證：兩種算法差多少、是否一致，處理完會在日誌裡統計。
# 只有邊界前後找得到明確的靜音／有聲交界才會動；連續語音、背景聲太大（建議開人聲分離）時原樣保留。
# 註：起訖點更貼近真正的聲音後，詞與詞之間量到的停頓會比以前短一點點（約 0.1 秒），
#     如果覺得 SPLIT_PAUSE_SEC 的拆分變少了，可以把它調小一點（例如 0.4）。
ALIGN_REFINE = True              # False = 只用 CTC 的結果（輸出檔名會加上 _ctc，方便跟微調版比較）
ALIGN_REFINE_BACK_SEC = 0.15     # 起點最多往前（提早）修正幾秒
ALIGN_REFINE_FWD_SEC = 0.06      # 起點最多往後（延遲）修正幾秒（CTC 起點落在還沒有聲音的地方時）
ALIGN_REFINE_END_FWD_SEC = 0.30  # 終點最多往後（延後）修正幾秒，找聲音真正結束的地方
ALIGN_REFINE_END_BACK_SEC = 0.08 # 終點最多往前（提早）修正幾秒（CTC 終點已經落在靜音裡時）
ALIGN_REFINE_MIN_CONTRAST_DB = 15.0  # 附近最大聲與背景至少差幾 dB 才判定；背景聲太大時放棄，維持 CTC 的時間
ALIGN_REFINE_THRESHOLD = 0.30    # 多大聲才算「有聲音」：在背景到最大聲之間（dB）的這個比例（0～1）。
                                 # 調大 = 保守（只認明顯的聲音，起點會比較晚、終點比較早，不容易把呼吸聲當成開頭）；調小 = 敏感
ALIGN_AGREE_SEC = 0.04           # CTC 與聲音的差距不超過這麼多秒，統計時算「一致」

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

# PyTorch 和 faster-whisper 各帶一份不同小版本的 cuDNN 9，放在同一個程序裡會出現
# 「Could not load symbol cudnnGetLibConfig. Error code 127」，所以對齊跟 Demucs 一樣
# 用獨立程序跑，主程式完全不 import torch。
ALIGN_WORKER = BASE_DIR / "align_worker.py"
