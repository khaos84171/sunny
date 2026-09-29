"""處理單一檔案的完整流程:（可選）人聲分離 → 轉錄 →（可選）對齊 → 拆分 → 輸出 SRT；以及只對齊現有 SRT。"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

from . import config, runtime
from .aligner import run_alignment
from .jobs import JOBS, JobCancelled
from .models import Word
from .separation import separate_vocals
from .splitter import _ensure_min_display, split_boundary_mode, split_segment
from .srt_io import add_leading_blank, format_timestamp, is_blank_text, parse_srt, write_srt
from .timing import apply_alignment, finalize_aligned_timing, report_alignment
from .transcribe import build_hotwords, get_model, looks_like_hallucination


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
            str(media_path), runtime.separation_work_dir, log_func,
            device=config.SEPARATION_DEVICE, model_name=config.SEPARATION_MODEL,
            segment=config.SEPARATION_SEGMENT,
            progress_func=lambda frac: overall_progress(config.SEPARATION_PROGRESS_SHARE * frac),
        )
        progress_func = lambda frac: overall_progress(
            config.SEPARATION_PROGRESS_SHARE + (1 - config.SEPARATION_PROGRESS_SHARE) * frac)
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
    align_tag = "_align" if config.ALIGN_REFINE else "_align_ctc"  # 關掉聲音微調的版本不會蓋掉微調版，方便比較
    output_path = os.path.join(runtime.output_dir, f"{srt_path.stem}{align_tag}.srt")
    write_srt(output_path, subs)
    progress_func(1.0)
    log_func(f"=== 完成，對齊後字幕已儲存：{output_path} ===\n")
    return output_path


def process_file(file_path: str, use_sep: bool, use_align: bool, use_split: bool,
                  hotword_list: list[str],
                  log_func, progress_func, add_blank: bool = True) -> str:
    """處理單一檔案：（可選）人聲分離 -> 轉錄 ->（可選）時間戳對齊 -> 輸出 SRT，回傳輸出檔案路徑。"""
    input_path = Path(file_path)
    base = input_path.stem  # 用原始檔名作為輸出檔名的基礎，取代固定的 "abc"

    sep_tag = "sep" if use_sep else "nosep"

    log_func(f"\n=== 開始處理：{input_path.name} ===")
    log_func(
        f"設定：模型 = {config.WHISPER_MODEL}、人聲分離 = {'是' if use_sep else '否'}、"
        f"時間戳對齊 = {'是' if use_align else '否'}、自動拆分 = {'是' if use_split else '否'}"
    )

    # 有做人聲分離時，分離佔進度條的前一段，轉錄與對齊縮進剩下的部分
    overall_progress = progress_func
    if use_sep:
        transcribe_input = separate_vocals(
            str(input_path), runtime.separation_work_dir, log_func,
            device=config.SEPARATION_DEVICE, model_name=config.SEPARATION_MODEL,
            segment=config.SEPARATION_SEGMENT,
            progress_func=lambda frac: overall_progress(config.SEPARATION_PROGRESS_SHARE * frac),
        )
        progress_func = lambda frac: overall_progress(
            config.SEPARATION_PROGRESS_SHARE + (1 - config.SEPARATION_PROGRESS_SHARE) * frac)
    else:
        transcribe_input = str(input_path)
    JOBS.check()

    model = get_model(log_func)

    # hotwords 由前端勾選傳入；每部影片出場的人不同，只放這次用得到的詞
    hotwords = build_hotwords(hotword_list)
    log_func(f"本次使用的 hotwords：{hotwords if hotwords else '（無）'}")

    segments, info = model.transcribe(
        transcribe_input,
        **config.TRANSCRIBE_OPTIONS,
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
        no_speech = getattr(segment, "no_speech_prob", None)
        if looks_like_hallucination(text, no_speech, getattr(segment, "avg_logprob", None)):
            shown = f"，無語音機率 {no_speech:.2f}" if no_speech is not None else ""
            log_func(f"[幻覺過濾] 略過 [{format_timestamp(segment.start)}] {text}{shown}")
            progress_func(min(segment.end / total, 1.0) * transcribe_weight)
            continue
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

    align_tag = ("_align" if config.ALIGN_REFINE else "_align_ctc") if align_result is not None else ""
    JOBS.check()
    split_tag = "" if use_split else "_nosplit"  # 有沒有拆分的版本不會互相覆蓋
    output_filename = f"{base}_{config.WHISPER_MODEL}_{sep_tag}{align_tag}{split_tag}.srt"
    output_path = os.path.join(runtime.output_dir, output_filename)
    write_srt(output_path, subs)

    progress_func(1.0)
    log_func(f"=== 完成，字幕已儲存：{output_path} ===\n")
    return output_path
