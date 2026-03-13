"""Vocal separation using Demucs."""

import subprocess
import sys
from pathlib import Path


def separate_vocals(audio_path: str, output_dir: str = None) -> Path:
    """Separate vocals from background music using Demucs.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory to save separated tracks. Defaults to ./separated.

    Returns:
        Path to the separated vocals file.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if output_dir is None:
        output_dir = audio_path.parent / "separated"

    output_dir = Path(output_dir)

    print(f"Separating vocals from: {audio_path.name}")
    print("This may take a few minutes...")

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-o", str(output_dir),
        str(audio_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed:\n{result.stderr}")

    stem_name = audio_path.stem
    vocals_path = output_dir / "htdemucs" / stem_name / "vocals.wav"

    if not vocals_path.exists():
        for model_dir in output_dir.iterdir():
            candidate = model_dir / stem_name / "vocals.wav"
            if candidate.exists():
                vocals_path = candidate
                break

    if not vocals_path.exists():
        raise FileNotFoundError(
            f"Vocals file not found. Check output in: {output_dir}"
        )

    print(f"Vocals saved to: {vocals_path}")
    return vocals_path
