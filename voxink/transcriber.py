"""Speech-to-text transcription using Whisper."""

from pathlib import Path

import whisper


def transcribe(audio_path: str, language: str = None, model_size: str = "medium") -> list[dict]:
    """Transcribe audio to text with timestamps using Whisper.

    Args:
        audio_path: Path to the audio file (preferably isolated vocals).
        language: Language code (e.g., "de", "en", "zh"). Auto-detected if None.
        model_size: Whisper model size: tiny, base, small, medium, large.

    Returns:
        List of segments, each with 'start', 'end', and 'text' keys.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    print(f"Transcribing: {audio_path.name}")
    options = {"word_timestamps": True}
    if language:
        options["language"] = language

    result = model.transcribe(str(audio_path), **options)

    detected_lang = result.get("language", "unknown")
    print(f"Detected language: {detected_lang}")

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        })

    print(f"Transcribed {len(segments)} segments.")
    return segments
