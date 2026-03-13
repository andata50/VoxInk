"""Vocal separation using Demucs."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def _find_ffmpeg() -> str | None:
    """Find ffmpeg executable path."""
    found = shutil.which("ffmpeg")
    if found:
        return found

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except ImportError:
        pass

    candidates = [
        Path("C:/ffmpeg-shared/bin/ffmpeg.exe"),
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path(sys.prefix) / "Scripts" / "ffmpeg.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    return None


def _load_audio(audio_path: Path, ffmpeg_exe: str | None) -> tuple[torch.Tensor, int]:
    """Load audio file as torch tensor. Convert to WAV first if needed."""
    path = audio_path

    # If not WAV, convert with ffmpeg
    if audio_path.suffix.lower() != ".wav":
        if not ffmpeg_exe:
            raise RuntimeError(
                "FFmpeg needed to convert non-WAV files. "
                "Install from https://ffmpeg.org/download.html"
            )
        wav_path = audio_path.with_suffix(".voxink_temp.wav")
        print(f"Converting to WAV: {audio_path.name}")
        cmd = [ffmpeg_exe, "-i", str(audio_path), "-ar", "44100", "-ac", "2", str(wav_path), "-y"]
        subprocess.run(cmd, capture_output=True, check=True)
        path = wav_path

    data, sr = sf.read(str(path), dtype="float32")

    # Clean up temp WAV
    if path != audio_path and path.exists():
        path.unlink()

    # soundfile returns (samples, channels), we need (channels, samples)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    else:
        data = data.T

    return torch.from_numpy(data), sr


def separate_vocals(audio_path: str, output_dir: str = None) -> Path:
    """Separate vocals from background music using Demucs.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory to save separated tracks. Defaults to ./separated.

    Returns:
        Path to the separated vocals file.
    """
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if output_dir is None:
        output_dir = audio_path.parent / "separated"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = _find_ffmpeg()

    print(f"Separating vocals from: {audio_path.name}")
    print("Loading audio...")
    wav, sr = _load_audio(audio_path, ffmpeg_exe)

    print("Loading Demucs model...")
    model = get_model("htdemucs")
    model.eval()

    # Demucs expects (batch, channels, samples) and specific sample rate
    if sr != model.samplerate:
        import torchaudio.functional as F
        wav = F.resample(wav, sr, model.samplerate)

    wav = wav.unsqueeze(0)  # Add batch dimension

    print("Separating... (this may take a few minutes)")
    with torch.no_grad():
        sources = apply_model(model, wav)

    # sources shape: (batch, num_sources, channels, samples)
    # Find vocals index
    source_names = model.sources
    vocals_idx = source_names.index("vocals")

    vocals = sources[0, vocals_idx].cpu().numpy()

    # Save vocals
    vocals_dir = output_dir / "htdemucs" / audio_path.stem
    vocals_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = vocals_dir / "vocals.wav"

    sf.write(str(vocals_path), vocals.T, model.samplerate)

    print(f"Vocals saved to: {vocals_path}")
    return vocals_path
