"""假的對齊程序（模擬 align_worker.py --serve 的通訊協定），用來測 AlignServer 的重試與錯誤處理。
模式由第一句字幕的文字決定：die_silently（第一個程序讀到工作就無聲結束）/ fail / crash / 其他 = 成功。"""
import json
import os
import sys

counter = os.environ["FAKE_COUNTER"]
n = int(open(counter).read() or 0) + 1 if os.path.exists(counter) else 1
open(counter, "w").write(str(n))
for line in sys.stdin:
    req = json.loads(line)
    mode = json.load(open(req["job"], encoding="utf-8"))["segments"][0]["text"]
    if mode == "die_silently" and n == 1:
        sys.exit(0)
    print("START", flush=True)
    if mode == "fail":
        print("ERROR boom 模擬", flush=True)
        print("END 2", flush=True)
        sys.exit(2)
    if mode == "crash":
        os._exit(9)
    json.dump({"spans": [[0.0, 1.0]], "confs": [0.9], "wide": [False], "word_spans": [None]}, open(req["result"], "w"))
    print(f"LOG 第 {n} 個程序做完了", flush=True)
    print("PROGRESS 10 10", flush=True)
    print("END 0", flush=True)
