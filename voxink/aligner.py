"""Align provided lyrics text to audio timestamps."""

from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


# --- Whisper transcription ---

def _get_whisper_segments(audio_path: str, language: str = None, model_size: str = "medium") -> list[dict]:
    """Get segment-level and word-level timestamps from faster-whisper."""
    print(f"Loading faster-whisper model: {model_size}")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"Transcribing for alignment: {Path(audio_path).name}")
    kwargs = {"word_timestamps": True, "vad_filter": True}
    if language:
        kwargs["language"] = language

    raw_segments, info = model.transcribe(audio_path, **kwargs)

    segments = []
    for seg in raw_segments:
        segments.append({
            "text": seg.text.strip(),
            "start": seg.start,
            "end": seg.end,
            "words": [
                {"text": w.word.strip(), "start": w.start, "end": w.end}
                for w in (seg.words or [])
            ],
        })

    print(f"faster-whisper transcribed {len(segments)} segments:")
    for s in segments:
        print(f"  [{s['start']:>7.2f} - {s['end']:>7.2f}] {s['text']}")

    return segments


# --- Audio energy / onset detection ---

def _detect_vocal_onsets(audio_path: str, hop_sec: float = 0.02, energy_threshold: float = 0.3) -> list[float]:
    """Detect vocal onset times using energy envelope."""
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)

    hop = int(sr * hop_sec)
    n_frames = len(data) // hop
    energy = np.array([
        np.sqrt(np.mean(data[i * hop:(i + 1) * hop] ** 2))
        for i in range(n_frames)
    ])

    if len(energy) == 0:
        return []

    max_e = energy.max()
    if max_e > 0:
        energy = energy / max_e

    onsets = []
    was_below = True
    min_gap = 0.5

    for i in range(1, len(energy)):
        if was_below and energy[i] >= energy_threshold:
            if not onsets or (i * hop_sec - onsets[-1]) >= min_gap:
                onsets.append(i * hop_sec)
            was_below = False
        elif energy[i] < energy_threshold * 0.6:
            was_below = True

    return onsets


# --- Fuzzy matching ---

def _similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


MATCH_THRESHOLD = 0.35


def _match_lyrics_to_segments(lines: list[str], whisper_segs: list[dict]) -> list[dict | None]:
    """Match each lyrics line to the best Whisper segment.

    Uses two passes:
    1. First find high-confidence matches (score >= 0.6) as anchors
    2. Then fill gaps with lower-threshold matches, constrained by anchors

    Returns a list parallel to lines, each entry is the matched whisper segment or None.
    """
    n_lines = len(lines)
    n_segs = len(whisper_segs)

    # Build a score matrix: lines x segments
    scores = []
    for line in lines:
        row = []
        for seg in whisper_segs:
            row.append(_similarity(line, seg["text"]))
        scores.append(row)

    matches = [None] * n_lines

    # --- Pass 1: High-confidence anchors only (strict threshold + forward order) ---
    HIGH_THRESHOLD = 0.6
    last_seg_idx = 0

    for i in range(n_lines):
        best_score = -1.0
        best_j = -1

        # Limit search window: don't jump more than 10 segments ahead
        search_end = min(n_segs, last_seg_idx + 10)
        for j in range(last_seg_idx, search_end):
            if scores[i][j] >= HIGH_THRESHOLD:
                # Take the first segment that meets the threshold.
                # This prevents repeated lyrics from jumping to a later
                # occurrence with a slightly higher score.
                best_score = scores[i][j]
                best_j = j
                break
            if scores[i][j] > best_score:
                best_score = scores[i][j]
                best_j = j

        if best_score >= HIGH_THRESHOLD and best_j >= 0:
            matches[i] = {
                "seg_idx": best_j,
                "score": best_score,
                "start": whisper_segs[best_j]["start"],
                "end": whisper_segs[best_j]["end"],
            }
            last_seg_idx = best_j + 1

    # --- Pass 2: Fill gaps between anchors with lower threshold ---
    anchor_indices = [i for i in range(n_lines) if matches[i] is not None]

    for gap_start_pos in range(len(anchor_indices) + 1):
        # Determine the segment range for this gap
        if gap_start_pos == 0:
            line_start = 0
            seg_range_start = 0
        else:
            prev_anchor = anchor_indices[gap_start_pos - 1]
            line_start = prev_anchor + 1
            seg_range_start = matches[prev_anchor]["seg_idx"] + 1

        if gap_start_pos < len(anchor_indices):
            next_anchor = anchor_indices[gap_start_pos]
            line_end = next_anchor
            seg_range_end = matches[next_anchor]["seg_idx"]
        else:
            line_end = n_lines
            seg_range_end = n_segs

        # Try to match unmatched lines within this constrained range
        cur_seg = seg_range_start
        for i in range(line_start, line_end):
            if matches[i] is not None:
                continue

            best_score = -1.0
            best_j = -1
            search_end = min(seg_range_end, cur_seg + 5)

            for j in range(cur_seg, search_end):
                if scores[i][j] > best_score:
                    best_score = scores[i][j]
                    best_j = j

            if best_score >= MATCH_THRESHOLD and best_j >= 0:
                matches[i] = {
                    "seg_idx": best_j,
                    "score": best_score,
                    "start": whisper_segs[best_j]["start"],
                    "end": whisper_segs[best_j]["end"],
                }
                cur_seg = best_j + 1

    return matches


def _match_line_to_words_in_segment(
    line: str, seg: dict, next_line: str | None = None
) -> tuple[float, float]:
    """Try to find a more precise start/end within a segment using word timestamps.

    If a Whisper segment contains multiple lyrics lines, use word matching to narrow down.
    When next_line is provided, avoids claiming boundary words that belong to the next line.
    """
    words = seg.get("words", [])
    if not words:
        return seg["start"], seg["end"]

    line_words = line.lower().split()
    if not line_words:
        return seg["start"], seg["end"]

    # Try to find the best matching position using exact line length (no padding)
    best_start_idx = 0
    best_score = -1
    for i in range(len(words)):
        end_i = min(i + len(line_words), len(words))
        candidate = " ".join(w["text"] for w in words[i:end_i]).lower()
        score = _similarity(" ".join(line_words), candidate)
        if score > best_score:
            best_score = score
            best_start_idx = i

    end_idx = min(best_start_idx + len(line_words), len(words)) - 1

    # Fix boundary overlap: if the next line starts with word(s) that also
    # appear at the end of our matched range, trim them so the repeated
    # word belongs to the next line.
    # e.g. "Observe well and see" / "See how it's meant to be"
    #   → don't let the second "see" be claimed by the first line
    if next_line and end_idx > best_start_idx:
        next_first = next_line.lower().split()[0].strip(".,!?;:'\"") if next_line.strip() else None
        if next_first:
            # Walk backwards from end_idx: trim consecutive words matching next_first
            while end_idx > best_start_idx:
                seg_word = words[end_idx]["text"].lower().strip(".,!?;:'\"")
                if seg_word == next_first:
                    end_idx -= 1
                else:
                    break

    return words[best_start_idx]["start"], words[end_idx]["end"]


# --- Onset-based fallback ---

def _estimate_time_from_onsets(
    onsets: list[float],
    prev_end: float,
) -> tuple[float, float]:
    """Find the next vocal onset after prev_end."""
    for onset in onsets:
        if onset >= prev_end + 0.3:
            return onset, onset + 3.0

    return prev_end + 0.5, prev_end + 3.5


# --- Main alignment ---

def align_lyrics(
    audio_path: str,
    lyrics_text: str,
    language: str = None,
    model_size: str = "medium",
) -> list[dict]:
    """Align provided lyrics to audio timestamps.

    Strategy:
    1. Whisper transcribes audio into segments (sentences + timestamps)
    2. Fuzzy-match each lyrics line to Whisper segments
    3. Matched lines get Whisper timestamps (anchors)
    4. Unmatched lines get interpolated between anchors or onset-detected

    Args:
        audio_path: Path to audio file (preferably isolated vocals).
        lyrics_text: Full lyrics text, one line per line.
        language: Language code. Auto-detected if None.
        model_size: Whisper model size.

    Returns:
        List of segments with 'start', 'end', and 'text' keys.
    """
    audio_path = str(audio_path)

    # Step A: Whisper transcription
    whisper_segs = _get_whisper_segments(audio_path, language, model_size)

    # Step B: Vocal onset detection
    print("Detecting vocal onsets...")
    onsets = _detect_vocal_onsets(audio_path)
    print(f"Found {len(onsets)} vocal onsets.")

    # Audio duration
    info = sf.info(audio_path)
    audio_duration = info.duration

    # Parse lyrics
    lines = [line.strip() for line in lyrics_text.strip().splitlines() if line.strip()]

    if not whisper_segs:
        print("Warning: Whisper detected no segments. Using onset-only alignment.")
        return _onset_only_alignment(lines, onsets, audio_duration)

    # Step C: Match lyrics lines to Whisper segments
    print(f"\nMatching {len(lines)} lyrics lines to {len(whisper_segs)} Whisper segments...")
    matches = _match_lyrics_to_segments(lines, whisper_segs)

    # Count anchors
    anchors = [(i, m) for i, m in enumerate(matches) if m is not None]
    print(f"Found {len(anchors)} anchor matches.")
    for i, m in anchors:
        print(f"  Line {i:>2}: [{m['start']:>7.2f}] \"{lines[i][:40]}\" ~ score={m['score']:.2f}")

    # Step D: Build final segments with interpolation
    segments = []

    for i, line in enumerate(lines):
        m = matches[i]

        if m is not None:
            # Anchored: use Whisper timestamp, try word-level precision
            wseg = whisper_segs[m["seg_idx"]]
            next_line = lines[i + 1] if i + 1 < len(lines) else None
            start, end = _match_line_to_words_in_segment(line, wseg, next_line)
            segments.append({"start": start, "end": end, "text": line})
        else:
            # Not matched: interpolate or use onsets
            # Find surrounding anchors
            prev_anchor = None
            next_anchor = None
            for ai, am in anchors:
                if ai < i:
                    prev_anchor = (ai, am)
                elif ai > i and next_anchor is None:
                    next_anchor = (ai, am)

            if prev_anchor and next_anchor:
                # Interpolate between anchors
                p_i, p_m = prev_anchor
                n_i, n_m = next_anchor
                gap = n_i - p_i
                if gap > 0:
                    t_range = n_m["start"] - p_m["end"]
                    t_per_line = t_range / gap
                    offset = i - p_i
                    start = p_m["end"] + t_per_line * (offset - 0.5)
                    end = p_m["end"] + t_per_line * (offset + 0.5)
                    start = max(start, p_m["end"])
                else:
                    prev_end = segments[-1]["end"] if segments else 0.0
                    start, end = _estimate_time_from_onsets(onsets, prev_end)
            elif prev_anchor:
                prev_end = segments[-1]["end"] if segments else prev_anchor[1]["end"]
                start, end = _estimate_time_from_onsets(onsets, prev_end)
            elif next_anchor:
                n_i, n_m = next_anchor
                lines_before = n_i - i
                start = max(0, n_m["start"] - 3.0 * lines_before)
                end = start + 3.0
            else:
                prev_end = segments[-1]["end"] if segments else 0.0
                start, end = _estimate_time_from_onsets(onsets, prev_end)

            segments.append({"start": start, "end": end, "text": line})

    # Ensure chronological order
    for i in range(1, len(segments)):
        if segments[i]["start"] < segments[i - 1]["start"]:
            segments[i]["start"] = segments[i - 1]["end"] + 0.1
        if segments[i]["end"] <= segments[i]["start"]:
            segments[i]["end"] = segments[i]["start"] + 2.0

    n_anchored = len(anchors)
    n_interpolated = len(segments) - n_anchored
    print(f"\nAligned {len(segments)} segments ({n_anchored} anchored, {n_interpolated} interpolated).")
    return segments


def _onset_only_alignment(lines: list[str], onsets: list[float], duration: float) -> list[dict]:
    """Fallback: align lyrics using only onset detection."""
    segments = []
    onset_idx = 0

    for line in lines:
        if onset_idx < len(onsets):
            start = onsets[onset_idx]
            onset_idx += 1
            end = onsets[onset_idx] if onset_idx < len(onsets) else start + 3.0
        else:
            start = segments[-1]["end"] + 0.5 if segments else 0.0
            end = start + 3.0

        segments.append({"start": start, "end": end, "text": line})

    print(f"Onset-only alignment: {len(segments)} segments.")
    return segments
