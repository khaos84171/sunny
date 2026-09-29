# 自動測試

不需要 GPU、不需要下載模型，也不需要真的裝 faster-whisper / torch / Demucs：這些都用 `stubs/`、`ml_stubs/`、`fakes/`
裡的假替身取代。

## 怎麼跑

```
pip install pytest numpy
python -m pytest tests
```

- 介面測試（`test_gui.py`）需要 tkinter 和顯示器，沒有的話會自動跳過。
  Linux 沒有螢幕時：`xvfb-run -a python -m pytest tests`。
- 想連斷詞（janome）的測試一起跑：`pip install janome`。沒裝的話「有 janome」那一組會跳過。
- 測試會在暫存資料夾裡工作，不會動到專案、也不會建立你設定的 `D:\桌面\whisper\…` 資料夾。

## 各檔案測什麼

| 檔案 | 內容 |
|---|---|
| `test_srt_io.py` | SRT 時間格式、各種編碼、多行字幕、開頭空白字幕 |
| `test_splitter.py` | 字幕拆分：句尾標點、停頓、長度上限、右括號、不切在詞中間（有／沒有 janome） |
| `test_timing.py` | 對齊後的時間微調（不推遲下一條）、套用對齊結果、檢查報告 |
| `test_dropped.py`, `test_settings_store.py` | 拖進來的東西整理、設定檔讀寫 |
| `test_jobs.py` | 取消與子程序管理 |
| `test_separation.py` | Demucs：呼叫方式、進度、快取（含殘檔）、取消、找 Demucs 的順序（用 `fakes/demucs`） |
| `test_transcribe.py` | GPU 偵測與 CPU 備援、Whisper 模型載入、幻覺片語過濾 |
| `test_pipeline.py` | 完整流程、進度條分配、輸出檔名、取消 |
| `test_align_worker.py` | `align_worker.py`：結果與重構前逐項一致（`golden/`）、常駐模式、顯存搬移、萬用字元、CTC |
| `test_aligner.py` | 主程式這一側的對齊程序管理（重用、逾時、取消、出錯、重試） |
| `test_runtime.py` | log 輪替、輸出資料夾與備用位置、啟動失敗的訊息框 |
| `test_gui.py` | 設定記憶、Hotwords 預設、日誌視窗、取消／佇列／拖放、錯誤回報、關閉視窗、DPI |
| `test_startup.py` | 入口 `w1_1.py` 的啟動保護（缺套件資料夾、缺必要套件、正常載入） |

## 黃金檔 `golden/`

`align_seed*_result.json` 與 `_stdout.txt` 是「拆檔／重構之前」的 `align_worker.py` 對同一個假模型工作的輸出。
新版的輸出必須跟它一致（數字容許極小的浮點誤差）。**只有在你是故意改變對齊演算法時**才重新產生這兩份檔案。

## 環境變數（只給測試用）

- `WHISPER_APP_NO_DIALOG=1`：啟動失敗時不跳出訊息框（否則自動測試會卡在訊息框）。
