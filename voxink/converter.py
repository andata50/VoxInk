"""Convert transcription segments to LRC and SRT formats."""

from pathlib import Path


def _format_lrc_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def _format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millis = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millis:03d}"


def segments_to_lrc(segments: list[dict], metadata: dict = None, word_level: bool = False) -> str:
    """Convert segments to LRC format string.

    If word_level=True and segments contain word timestamps, generates
    enhanced LRC with per-word timing: [mm:ss.xx]<mm:ss.xx>word1 <mm:ss.xx>word2 ...
    """
    lines = []

    if metadata:
        tag_map = {"title": "ti", "artist": "ar", "album": "al"}
        for key, tag in tag_map.items():
            if key in metadata:
                lines.append(f"[{tag}:{metadata[key]}]")
        lines.append("[tool:VoxInk]")
        lines.append("")

    for seg in segments:
        time_tag = _format_lrc_time(seg["start"])
        words = seg.get("words", [])

        if word_level and words:
            word_parts = []
            for w in words:
                wt = _format_lrc_time(w["start"])
                word_parts.append(f"<{wt}>{w['text']}")
            lines.append(f"[{time_tag}]{' '.join(word_parts)}")
        else:
            lines.append(f"[{time_tag}]{seg['text']}")

    return "\n".join(lines)


def segments_to_srt(segments: list[dict]) -> str:
    """Convert segments to SRT subtitle format."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_srt_time(seg["start"])
        end = _format_srt_time(seg["end"])
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"])
        lines.append("")

    return "\n".join(lines)


def save_lrc(segments: list[dict], output_path: str, metadata: dict = None, word_level: bool = False):
    output_path = Path(output_path)
    content = segments_to_lrc(segments, metadata, word_level=word_level)
    output_path.write_text(content, encoding="utf-8")
    print(f"LRC saved to: {output_path}")


def save_srt(segments: list[dict], output_path: str):
    output_path = Path(output_path)
    content = segments_to_srt(segments)
    output_path.write_text(content, encoding="utf-8")
    print(f"SRT saved to: {output_path}")
