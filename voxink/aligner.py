"""Align provided lyrics text to audio timestamps."""

from difflib import SequenceMatcher
from pathlib import Path

import whisper


def _get_word_timestamps(audio_path: str, language: str = None, model_size: str = "medium") -> list[dict]:
    """Get word-level timestamps from Whisper."""
    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    print(f"Transcribing for alignment: {Path(audio_path).name}")
    options = {"word_timestamps": True}
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)

    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({
                "text": w["word"].strip(),
                "start": w["start"],
                "end": w["end"],
            })

    return words


def _similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _find_best_match(line_words: list[str], whisper_words: list[dict], search_start: int) -> tuple[int, int]:
    """Find the best matching position for a line in the whisper word list.

    Returns (start_idx, end_idx) in whisper_words.
    """
    line_text = " ".join(line_words).lower()
    n = len(line_words)
    best_score = -1
    best_start = search_start
    best_end = search_start + n

    # Search window: from search_start to reasonable end
    search_end = min(len(whisper_words), search_start + n * 3 + 10)

    for i in range(search_start, search_end):
        for length in range(max(1, n - 2), n + 3):
            j = i + length
            if j > len(whisper_words):
                break
            candidate = " ".join(w["text"] for w in whisper_words[i:j]).lower()
            score = _similarity(line_text, candidate)
            if score > best_score:
                best_score = score
                best_start = i
                best_end = j

    return best_start, best_end


def align_lyrics(
    audio_path: str,
    lyrics_text: str,
    language: str = None,
    model_size: str = "medium",
) -> list[dict]:
    """Align provided lyrics to audio timestamps.

    Args:
        audio_path: Path to audio file (preferably isolated vocals).
        lyrics_text: Full lyrics text, one line per line.
        language: Language code. Auto-detected if None.
        model_size: Whisper model size.

    Returns:
        List of segments with 'start', 'end', and 'text' keys.
    """
    audio_path = str(audio_path)

    # Get word-level timestamps from Whisper
    whisper_words = _get_word_timestamps(audio_path, language, model_size)

    if not whisper_words:
        print("Warning: Whisper detected no words in the audio.")
        return []

    # Parse lyrics into lines
    lines = [line.strip() for line in lyrics_text.strip().splitlines() if line.strip()]

    print(f"Aligning {len(lines)} lyrics lines to {len(whisper_words)} detected words...")

    segments = []
    search_pos = 0

    for line in lines:
        line_words = line.split()
        if not line_words:
            continue

        start_idx, end_idx = _find_best_match(line_words, whisper_words, search_pos)

        if start_idx < len(whisper_words) and end_idx <= len(whisper_words):
            seg = {
                "start": whisper_words[start_idx]["start"],
                "end": whisper_words[end_idx - 1]["end"],
                "text": line,  # Use the user's original text
            }
            segments.append(seg)
            search_pos = end_idx  # Move forward for next line
        else:
            # Fallback: estimate based on position
            if segments:
                last_end = segments[-1]["end"]
                seg = {
                    "start": last_end + 0.5,
                    "end": last_end + 3.0,
                    "text": line,
                }
                segments.append(seg)
                search_pos = end_idx

    print(f"Aligned {len(segments)} segments.")
    return segments
