"""Command-line interface for VoxInk."""

import argparse
from pathlib import Path

from . import __version__
from .separator import separate_vocals
from .transcriber import transcribe
from .converter import save_lrc, save_srt


def main():
    parser = argparse.ArgumentParser(
        prog="voxink",
        description="VoxInk - Extract timed lyrics from songs automatically.",
    )
    parser.add_argument("audio", help="Path to the audio file (mp3, wav, flac, etc.)")
    parser.add_argument("-l", "--language", default=None,
                        help="Language code (e.g., de, en, zh). Auto-detected if omitted.")
    parser.add_argument("-m", "--model", default="medium",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("-f", "--format", default="lrc",
                        choices=["lrc", "srt", "both"],
                        help="Output format (default: lrc)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file path (default: same name as input)")
    parser.add_argument("--skip-separation", action="store_true",
                        help="Skip vocal separation, transcribe audio directly")
    parser.add_argument("--vocals-only", action="store_true",
                        help="Only separate vocals, skip transcription")
    parser.add_argument("--title", default=None, help="Song title for LRC metadata")
    parser.add_argument("--artist", default=None, help="Artist name for LRC metadata")
    parser.add_argument("-V", "--version", action="version", version=f"VoxInk {__version__}")

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        return 1

    # Step 1: Vocal separation
    if args.skip_separation:
        vocals_path = audio_path
        print("Skipping vocal separation.")
    else:
        print("=== Step 1: Separating vocals ===")
        vocals_path = separate_vocals(str(audio_path))

    if args.vocals_only:
        print("Done. Vocals separated.")
        return 0

    # Step 2: Transcribe
    print("\n=== Step 2: Transcribing lyrics ===")
    segments = transcribe(str(vocals_path), language=args.language, model_size=args.model)

    if not segments:
        print("No lyrics detected.")
        return 1

    # Step 3: Save output
    print("\n=== Step 3: Saving output ===")
    output_base = Path(args.output) if args.output else audio_path.with_suffix("")

    metadata = {}
    if args.title:
        metadata["title"] = args.title
    elif audio_path.stem:
        metadata["title"] = audio_path.stem
    if args.artist:
        metadata["artist"] = args.artist

    if args.format in ("lrc", "both"):
        save_lrc(segments, str(output_base.with_suffix(".lrc")), metadata or None)

    if args.format in ("srt", "both"):
        save_srt(segments, str(output_base.with_suffix(".srt")))

    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
