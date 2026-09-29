"""
量對齊準確度：拿「人工校正過的 SRT」當標準答案，量程式輸出的字幕起點、終點差多少秒。

用法（在程式資料夾裡開命令提示字元）：
    python -m whisper_app.evaluate 人工校正.srt 程式輸出.srt
    python -m whisper_app.evaluate 人工校正.srt 微調版.srt CTC版.srt     ← 同一份標準答案比較好幾個版本
可用選項：
    --folds 5         交叉驗證分幾折（預設 5）
    --tolerance 0.04 0.08 0.12    「準確率」用的容許誤差（秒）
    --show 10         列出最不準的幾條（預設 10）
    --json 結果.json   把結果存成 JSON

報告內容：
    ・配對：用文字相似度把標準答案的每一條找到輸出裡對應的那一條（拆分方式不同的字幕會配不到）
    ・誤差：輸出 - 標準答案（正值 = 輸出比較晚）。起點與終點分開算：
      平均／中位數／90%／最大誤差、落在各容許誤差內的比例（＝準確率）、系統性偏差（有號中位數）
    ・交叉驗證：把配對好的字幕依時間順序分成 K 折，每次用其中 K-1 折算出「固定偏差」，
      拿去修正沒看過的那一折，看修正後的誤差有沒有真的變小——這樣得到的改善是沒有作弊的
      （不是拿同一批資料又調參數又打分數），可以據以決定要不要調整 ALIGN_START_LEAD_SEC／ALIGN_END_HOLD_SEC
"""

from __future__ import annotations

import argparse
import bisect
import difflib
import json
import statistics
import sys
import unicodedata
from pathlib import Path

from . import config
from .srt_io import format_timestamp, is_blank_text, parse_srt

DEFAULT_TOLERANCES = (0.04, 0.08, 0.12)


def normalize_text(text: str) -> str:
    """比對用：全形半形統一、轉小寫，拿掉空白、標點與符號。"""
    kept = []
    for ch in unicodedata.normalize("NFKC", text).lower():
        if unicodedata.category(ch)[0] in "LN":
            kept.append(ch)
    return "".join(kept)


def load_subtitles(path) -> list[dict]:
    """讀 SRT，拿掉看不到字的字幕（例如開頭空白字幕）與沒有實際文字的字幕，依起點排序。"""
    subs = [s for s in parse_srt(path) if not is_blank_text(s["text"]) and normalize_text(s["text"])]
    subs.sort(key=lambda s: (s["start"], s["end"]))
    return subs


def match_subtitles(ref: list[dict], pred: list[dict], max_shift: float = 3.0, min_similarity: float = 0.8):
    """
    把標準答案的每一條配到輸出裡文字最像的一條（起點差不超過 max_shift 秒，而且順序不能倒過來）。
    回傳 [(標準答案編號, 輸出編號, 相似度), ...]，編號從 0 起算。
    """
    ref_text = [normalize_text(s["text"]) for s in ref]
    pred_text = [normalize_text(s["text"]) for s in pred]
    starts = [s["start"] for s in pred]
    pairs, last = [], -1
    for i, r in enumerate(ref):
        lo = bisect.bisect_left(starts, r["start"] - max_shift)
        hi = bisect.bisect_right(starts, r["start"] + max_shift)
        best = None
        for j in range(max(lo, last + 1), hi):
            sim = difflib.SequenceMatcher(None, ref_text[i], pred_text[j], autojunk=False).ratio()
            key = (round(sim, 3), -abs(starts[j] - r["start"]))
            if best is None or key > best[0]:
                best = (key, j, sim)
        if best is not None and best[2] >= min_similarity:
            pairs.append((i, best[1], best[2]))
            last = best[1]
    return pairs


def boundary_errors(ref: list[dict], pred: list[dict], pairs) -> list[dict]:
    """配對好的每一條：起點誤差、終點誤差（輸出 - 標準答案，單位秒）。"""
    return [{"ref": i, "pred": j, "start": pred[j]["start"] - ref[i]["start"], "end": pred[j]["end"] - ref[i]["end"]}
            for i, j, _ in pairs]


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def summarize(errors: list[float], tolerances=DEFAULT_TOLERANCES) -> dict:
    """一組有號誤差（秒）的統計。"""
    n = len(errors)
    if n == 0:
        return {"n": 0}
    absolute = sorted(abs(e) for e in errors)
    return {
        "n": n,
        "mean_abs": sum(absolute) / n,
        "median_abs": _percentile(absolute, 0.5),
        "p90": _percentile(absolute, 0.9),
        "max": absolute[-1],
        "bias": statistics.median(errors),          # 有號中位數：系統性偏差
        "within": {tol: sum(1 for a in absolute if a <= tol) / n for tol in tolerances},
    }


def cross_validate_offset(errors: list[float], folds: int = 5, tolerances=DEFAULT_TOLERANCES) -> dict | None:
    """
    K 折交叉驗證「加一個固定偏移量」有沒有用。依時間順序把誤差分成連續的 K 折；
    每次用其他 K-1 折的中位數當偏移量，修正被留下來的那一折，最後合併所有折修正後的誤差統計。
    資料太少（每折不到 2 個）回傳 None。
    """
    n = len(errors)
    folds = min(folds, n // 2)
    if folds < 2:
        return None
    bounds = [round(n * k / folds) for k in range(folds + 1)]
    residuals, offsets = [], []
    for k in range(folds):
        train = errors[:bounds[k]] + errors[bounds[k + 1]:]
        offset = statistics.median(train)
        offsets.append(offset)
        residuals.extend(e - offset for e in errors[bounds[k]:bounds[k + 1]])
    return {"folds": folds, "offsets": offsets, "after": summarize(residuals, tolerances),
            "before": summarize(errors, tolerances)}


def evaluate(ref_path, pred_path, tolerances=DEFAULT_TOLERANCES, folds: int = 5,
             max_shift: float = 3.0, min_similarity: float = 0.8) -> dict:
    """比較一份輸出與標準答案，回傳所有統計（可存成 JSON）。"""
    ref, pred = load_subtitles(ref_path), load_subtitles(pred_path)
    pairs = match_subtitles(ref, pred, max_shift, min_similarity)
    errors = boundary_errors(ref, pred, pairs)
    result = {
        "ref_path": str(ref_path), "pred_path": str(pred_path),
        "ref_count": len(ref), "pred_count": len(pred), "matched": len(pairs),
        "start": summarize([e["start"] for e in errors], tolerances),
        "end": summarize([e["end"] for e in errors], tolerances),
        "cv_start": cross_validate_offset([e["start"] for e in errors], folds, tolerances),
        "cv_end": cross_validate_offset([e["end"] for e in errors], folds, tolerances),
        "worst": [],
    }
    worst = sorted(errors, key=lambda e: max(abs(e["start"]), abs(e["end"])), reverse=True)
    for e in worst:
        r = ref[e["ref"]]
        result["worst"].append({"ref_time": r["start"], "text": r["text"].replace("\n", " "),
                                "start_error": e["start"], "end_error": e["end"]})
    return result


# =========================================================
# === 文字報告 ===
# =========================================================
def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _row(label: str, s: dict, tolerances) -> str:
    if not s["n"]:
        return f"  {label}   （沒有配對到的字幕）"
    within = "  ".join(_pct(s["within"][t]) for t in tolerances)
    direction = "輸出偏晚" if s["bias"] > 0 else "輸出偏早"
    return (f"  {label}   {s['mean_abs']:6.3f}   {s['median_abs']:6.3f}  {s['p90']:6.3f}  {s['max']:6.3f}   "
            f"{within}   {s['bias']:+.3f}（{direction}）")


def format_report(result: dict, tolerances=DEFAULT_TOLERANCES, show: int = 10) -> str:
    lines = [f"=== {Path(result['pred_path']).name}  對照標準答案  {Path(result['ref_path']).name} ===",
             f"標準答案 {result['ref_count']} 條，輸出 {result['pred_count']} 條，配對成功 {result['matched']} 條"
             f"（沒配到：標準答案 {result['ref_count'] - result['matched']} 條、輸出 {result['pred_count'] - result['matched']} 條；"
             "拆分方式不同或文字被改過的字幕會配不到，不列入統計）"]
    if not result["matched"]:
        lines.append("沒有任何一條配對成功，無法統計。請確認兩份 SRT 是同一部影片、文字大致相同。")
        return "\n".join(lines)

    head = "  ".join(f"<={t:g}秒" for t in tolerances)
    lines += ["", "誤差 = 輸出 - 標準答案（單位：秒）；右邊的百分比 = 誤差在該容許範圍內的字幕比例（準確率）",
              f"         平均絕對  中位數    90%    最大    {head}    系統性偏差(有號中位數)",
              _row("起點", result["start"], tolerances), _row("終點", result["end"], tolerances)]

    lines += ["", "交叉驗證（依時間順序分折；用其他折算出固定偏移量，修正沒看過的那一折）："]
    for label, key, setting in (("起點", "cv_start", "ALIGN_START_LEAD_SEC"), ("終點", "cv_end", "ALIGN_END_HOLD_SEC")):
        cv = result[key]
        if cv is None:
            lines.append(f"  {label}：配對到的字幕太少，做不了交叉驗證")
            continue
        before, after = cv["before"], cv["after"]
        gain = before["mean_abs"] - after["mean_abs"]
        verdict = (f"有幫助，平均誤差 {before['mean_abs']:.3f} → {after['mean_abs']:.3f} 秒"
                   f"（{cv['folds']} 折，各折偏移量 {min(cv['offsets']):+.3f}～{max(cv['offsets']):+.3f}）"
                   if gain >= 0.003 else "固定偏移量沒有明顯幫助（誤差不是系統性的偏早或偏晚）")
        within = "  ".join(f"<={t:g}秒 {_pct(after['within'][t])}" for t in tolerances)
        lines.append(f"  {label}：{verdict}")
        if gain >= 0.003:
            lines.append(f"        修正後準確率：{within}")
            bias = result["start" if key == "cv_start" else "end"]["bias"]
            if key == "cv_start":
                new = config.ALIGN_START_LEAD_SEC + bias
                lines.append(f"        → 建議 {setting} 從 {config.ALIGN_START_LEAD_SEC:g} 改成 {new:.2f}"
                             "（字幕比第一個字提早幾秒出現；負值 = 晚一點）")
            else:
                new = max(config.ALIGN_END_HOLD_SEC - bias, 0.0)
                lines.append(f"        → 建議 {setting} 從 {config.ALIGN_END_HOLD_SEC:g} 改成 {new:.2f}"
                             "（只影響後面有空隙的字幕；後面緊接下一條的不受影響）")

    if show and result["worst"]:
        lines += ["", f"誤差最大的 {min(show, len(result['worst']))} 條（標準答案的時間 / 起點誤差 / 終點誤差 / 文字）："]
        for w in result["worst"][:show]:
            lines.append(f"  [{format_timestamp(w['ref_time'])}] {w['start_error']:+.3f}  {w['end_error']:+.3f}  {w['text']}")
    return "\n".join(lines)


def format_comparison(results: list[dict], tolerances=DEFAULT_TOLERANCES) -> str:
    """好幾個輸出對照同一份標準答案時的總表。"""
    tol = tolerances[1] if len(tolerances) > 1 else tolerances[0]
    lines = ["=== 總表（越小越好；準確率越大越好）===",
             f"  檔案                                   配對   起點平均  起點<={tol:g}  終點平均  終點<={tol:g}"]
    for r in results:
        if not r["matched"]:
            lines.append(f"  {Path(r['pred_path']).name:<36}   0")
            continue
        s, e = r["start"], r["end"]
        lines.append(f"  {Path(r['pred_path']).name:<36} {r['matched']:5d}   {s['mean_abs']:6.3f}  {_pct(s['within'][tol])}"
                     f"   {e['mean_abs']:6.3f}  {_pct(e['within'][tol])}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m whisper_app.evaluate", description="用人工校正過的 SRT 量對齊準確度")
    parser.add_argument("reference", help="人工校正過的 SRT（標準答案）")
    parser.add_argument("predicted", nargs="+", help="要評分的輸出 SRT（可以給好幾個來比較）")
    parser.add_argument("--folds", type=int, default=5, help="交叉驗證分幾折（預設 5）")
    parser.add_argument("--tolerance", type=float, nargs="+", default=list(DEFAULT_TOLERANCES), help="準確率用的容許誤差（秒）")
    parser.add_argument("--show", type=int, default=10, help="列出最不準的幾條（預設 10）")
    parser.add_argument("--max-shift", type=float, default=3.0, help="配對時起點最多差幾秒（預設 3）")
    parser.add_argument("--min-similarity", type=float, default=0.8, help="配對要求的文字相似度（預設 0.8）")
    parser.add_argument("--json", help="把結果存成 JSON 檔")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # 主控台編碼印不出的字用 ? 代替，不要整個中斷
    tolerances = tuple(sorted(args.tolerance))
    results = []
    for pred in args.predicted:
        for path in (args.reference, pred):
            if not Path(path).is_file():
                print(f"找不到檔案：{path}", file=sys.stderr)
                return 2
        result = evaluate(args.reference, pred, tolerances, args.folds, args.max_shift, args.min_similarity)
        results.append(result)
        print(format_report(result, tolerances, args.show))
        print()
    if len(results) > 1:
        print(format_comparison(results, tolerances))
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n結果已存成 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
