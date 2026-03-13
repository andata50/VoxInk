"""Shared utility to find and configure ffmpeg."""

import os
import shutil
import sys
import tempfile
from pathlib import Path


def find_ffmpeg() -> str | None:
    """Find ffmpeg executable path (must be named 'ffmpeg' or 'ffmpeg.exe')."""
    found = shutil.which("ffmpeg")
    if found:
        return found

    # Check common install locations first (properly named binaries)
    candidates = [
        Path("C:/ffmpeg-shared/bin/ffmpeg.exe"),
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path(sys.prefix) / "Scripts" / "ffmpeg.exe",
        Path.home() / "AppData" / "Roaming" / "Python" / "Python312" / "Scripts" / "ffmpeg.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    # Fallback: imageio_ffmpeg (may have non-standard name)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except ImportError:
        pass

    return None


def ensure_ffmpeg_in_path():
    """Ensure an ffmpeg binary named 'ffmpeg' is available in PATH."""
    if shutil.which("ffmpeg"):
        return

    exe = find_ffmpeg()
    if not exe:
        raise RuntimeError(
            "FFmpeg not found. Install it:\n"
            "  pip install imageio-ffmpeg\n"
            "  or download from https://ffmpeg.org/download.html"
        )

    exe_path = Path(exe)

    # If the binary isn't named 'ffmpeg[.exe]', create a symlink/copy
    if exe_path.stem.lower() not in ("ffmpeg", "ffmpeg.exe"):
        link_dir = Path(tempfile.mkdtemp(prefix="voxink_ffmpeg_"))
        if sys.platform == "win32":
            link_path = link_dir / "ffmpeg.exe"
        else:
            link_path = link_dir / "ffmpeg"

        shutil.copy2(str(exe_path), str(link_path))
        ffmpeg_dir = str(link_dir)
    else:
        ffmpeg_dir = str(exe_path.parent)

    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    print(f"Found ffmpeg in: {ffmpeg_dir}")
