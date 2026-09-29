"""
Whisper 字幕產生器（拖放視窗版／可雙擊啟動）
— 介面採 QuizKnock 風格配色（白／紅／黑）—

檔案配置（要放在同一個資料夾）：
    w1_1.py           入口（就是這個檔案）。雙擊「字幕產生器.vbs」或執行 python w1_1.py 都是從這裡開始。
    align_worker.py   時間戳對齊用的獨立程序。
    whisper_app/      程式本體。想調整設定（路徑、各種門檻、hotwords 候選詞…）改 whisper_app/config.py。
    tests/            自動測試（開發用，不影響使用；執行方式見 tests/README.md）。

用法：
    1. 安裝額外套件：pip install tkinterdnd2 transformers janome
       （transformers 是「時間戳對齊」功能用的；沒裝也能跑，只是會自動略過對齊）
       （janome 是日文斷詞，讓「自動拆分」只在詞與詞之間切，不會把「熊本」切成「熊／本」；
         沒裝也能跑，會改用比較粗略的判斷規則）
    2. 平常用「字幕產生器.vbs」雙擊開啟即可（不會跳出黑色 cmd 視窗）。
       也可以直接執行 python w1_1.py 來測試。
       .vbs 會自己找電腦上能用的 Python（3.9 以上）：先看 PATH，再看 py launcher 和登錄檔（沒勾
       「Add to PATH」也找得到），Microsoft Store 的空殼捷徑排最後，而且每個都會先試跑過。
       想指定某個 Python 或虛擬環境，改 .vbs 開頭的 PYTHONW_PATH。程式異常結束時 .vbs 會跳出訊息，
       提示去看 whisper_app.log。同時只能開一個視窗（再開一次會提示已經開著；要允許多開，
       把 whisper_app/config.py 的 ALLOW_MULTIPLE_INSTANCES 改成 True）。
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
       faster-whisper 的 cuDNN 版本衝突），請務必把它們放在一起。
    7. 只想重新對齊現有字幕、不重跑 Whisper 時：把 .srt 和對應的影片「一起」
       拖進拖放區即可（例如手動改過字幕文字之後）。只拖 .srt 的話會跳出視窗
       讓你選對應的影片。輸出為原檔名加上 _align，不會覆蓋原本的 .srt。
    8. 勾選「自動拆分」時，Whisper 會多輸出每個詞的時間（word_timestamps），
       再把一條裡塞了好幾句的字幕，依「句尾標點 → 詞與詞之間的停頓 → 長度上限」
       拆成多條。有勾「時間戳對齊」時，會先拿 Whisper 原本的整段去對齊（文字長
       對得穩），align_worker.py 再傳回 wav2vec2 算出的每個詞的時間，用這個時間
       判斷停頓、決定在哪裡拆——比 Whisper 自己估的詞時間準很多。沒勾對齊時就用
       Whisper 的詞時間來拆。開啟後轉錄會稍微慢一點。門檻在 whisper_app/config.py 的「字幕拆分」設定區。
    9. 勾選「在最前面加一條空白字幕」時，輸出的 .srt 第一條會是從 00:00:00,000 開始、
       到第一句字幕出現為止的空白字幕。有些剪輯軟體匯入 SRT 時會把第一條字幕放在
       播放頭的位置，有這條空白字幕墊在 0 秒，把播放頭放在影片開頭再匯入就能對齊。
       只對齊模式也適用；輸入的 .srt 裡原本就有的空白字幕會先拿掉再重新加，不會變兩條。
   10. 需要 Python 3.9 以上。輸出資料夾（OUTPUT_DIR）和人聲分離的暫存資料夾（SEPARATION_WORK_DIR，都在 whisper_app/config.py）
       預設在 D:\桌面\whisper 底下；建立不起來（例如這台電腦沒有 D 槽）時，會自動改用腳本旁邊的
       text／separated 資料夾，實際用到哪裡會寫在視窗開啟後日誌的第一行。
   11. 人聲分離用獨立程序跑 Demucs（不會跳出黑色視窗，進度會顯示在進度條和日誌裡）。結果存在
       separation_work_dir 底下，同一個檔案（大小與修改時間沒變）再處理時直接重用。
   12. 右邊的「取消處理」會停掉目前的檔案（連同背景的 Demucs／對齊程序）並清掉排隊中的；
       處理中直接關閉視窗會先詢問，確定關閉時一樣會把背景程序停掉。
       也可以把資料夾拖進來：會把裡面（含子資料夾）的影片／音訊依檔名加入佇列，
       不是影片／音訊的檔案會略過並在日誌說明。認得的副檔名在 MEDIA_EXTENSIONS。
   13. 連續處理多個檔案時，對齊模型（wav2vec2）只載入一次：對齊程序會留著給下一個檔案用
       （ALIGN_KEEP_ALIVE_SEC 秒沒用到就自己結束），每個檔案做完先把模型搬回 CPU，顯存留給 Whisper／Demucs。
       Whisper／Demucs／對齊的裝置預設 "auto"：沒有 NVIDIA GPU 就自動改用 CPU（會慢很多）。
       轉錄時會略過 Whisper 在靜音或背景音樂上編出來的固定片語（例如「ご視聴ありがとうございました」），
       被略過的每一筆都寫在日誌，不需要時把 FILTER_HALLUCINATIONS 改成 False。
       沒有勾「自動拆分」時，輸出檔名結尾會多一個 _nosplit，不會蓋掉有拆分的版本。
   14. 視窗上的勾選狀態和 Hotwords 會記在腳本旁邊的 whisper_settings.json，下次開啟沿用；
       第一次開啟時「人名」都不勾、「通用詞」預設勾選（人名每部影片不同，不相關的反而會干擾辨識）。
       whisper_app.log 超過 2 MB 會換檔（只留最近 3 份舊的），日誌視窗最多留 LOG_MAX_LINES 行，
       往上捲動看舊訊息時不會被新訊息拉回底部。螢幕縮放大於 100% 時視窗不再被系統放大而變模糊。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))  # 雙擊啟動時通常已經在裡面，保險起見


def _fatal_before_logging(message: str) -> None:
    """連 whisper_app 資料夾都載不進來時的最後手段：留一個說明檔，並跳出 Windows 訊息框。"""
    import traceback
    detail = traceback.format_exc()
    try:
        (BASE_DIR / "whisper_startup_error.txt").write_text(message + "\n\n" + detail, encoding="utf-8")
    except OSError:
        pass
    if os.environ.get("WHISPER_APP_NO_DIALOG"):  # 自動測試用：只留說明檔，不要跳出會卡住的訊息框
        return
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, message + "\n\n" + detail[-600:], "Whisper 字幕產生器 - 發生錯誤", 0x10)
    except Exception:
        pass


try:
    from whisper_app import runtime  # 只用標準函式庫，一定載得進來（除非資料夾不見了）
except Exception:
    _fatal_before_logging("找不到 whisper_app 資料夾，或它不完整。請把整個 whisper_app 資料夾跟 w1_1.py 放在一起。")
    raise

runtime.setup_logging()  # 最先做：之後任何錯誤、print 都會進 log 檔

if not runtime.acquire_single_instance():  # 已經有另一個視窗在執行：不要再開一個
    runtime.show_notice("Whisper 字幕產生器已經開著了（請看工作列）。\n同時只能開一個視窗。")
    sys.exit(0)

try:
    from whisper_app import gui  # 這一步才會載入 tkinterdnd2、faster_whisper 等
except Exception:
    runtime.show_fatal_error("缺少必要套件或載入失敗（例如 tkinterdnd2 / faster_whisper 沒裝好）。\n\n" + runtime.python_hint())
    raise

runtime.setup_dirs()  # 建立輸出資料夾；失敗時會自己跳訊息框並丟出例外


def main() -> None:
    gui.run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        runtime.show_fatal_error("應用程式啟動或執行時發生未預期的錯誤。")
