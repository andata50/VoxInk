"""Speech-to-text transcription using faster-whisper."""

from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


def _detect_onsets(audio_path: str, hop_sec: float = 0.01) -> list[float]:
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
    threshold = 0.15

    for i in range(1, len(energy)):
        if was_below and energy[i] >= threshold:
            if not onsets or (i * hop_sec - onsets[-1]) >= 0.3:
                onsets.append(i * hop_sec)
            was_below = False
        elif energy[i] < threshold * 0.5:
            was_below = True

    return onsets


def _snap_to_onset(start: float, onsets: list[float], max_early: float = 1.0) -> float:
    """If a vocal onset exists slightly before the Whisper timestamp, use it."""
    best = start
    for onset in onsets:
        if onset > start:
            break
        if start - onset <= max_early:
            best = onset
    return best


def transcribe(audio_path: str, language: str = None, model_size: str = "medium") -> list[dict]:
    """Transcribe audio to text with timestamps using faster-whisper.

    Args:
        audio_path: Path to the audio file (preferably isolated vocals).
        language: Language code (e.g., "de", "en", "zh"). Auto-detected if None.
        model_size: Whisper model size: tiny, base, small, medium, large-v3.

    Returns:
        List of segments, each with 'start', 'end', and 'text' keys.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Loading faster-whisper model: {model_size}")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"Transcribing: {audio_path.name}")
    kwargs = {"word_timestamps": True, "vad_filter": True}
    if language:
        kwargs["language"] = language

    raw_segments, info = model.transcribe(str(audio_path), **kwargs)

    print(f"Detected language: {info.language} (prob={info.language_probability:.2f})")

    # Detect vocal onsets for timestamp correction
    onsets = _detect_onsets(str(audio_path))

    segments = []
    for seg in raw_segments:
        words = [
            {"text": w.word.strip(), "start": w.start, "end": w.end}
            for w in (seg.words or [])
        ]
        # Use first word's timestamp for more precise line start
        if words:
            start = words[0]["start"]
        else:
            start = seg.start
        # Further correct with onset detection
        start = _snap_to_onset(start, onsets)
        segments.append({
            "start": start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": words,
        })

    print(f"Transcribed {len(segments)} segments.")
    return segments
