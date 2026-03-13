---
title: VoxInk
emoji: "\U0001F3B5"
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.29.1"
app_file: app.py
pinned: false
license: mit
---

# VoxInk

> Extract timed lyrics from songs automatically.

VoxInk separates vocals from music, transcribes the lyrics with timestamps, and generates LRC/SRT files — all in one command.

**Try it online (no install needed):** [VoxInk on Hugging Face Spaces](https://huggingface.co/spaces/andata50/VoxInk)

## How it works

```
Audio File → [Demucs] → Isolated Vocals → [Whisper] → Timed Lyrics → .lrc / .srt
```

## Installation

### Prerequisites

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/download.html) (required by both Demucs and Whisper)

### Install VoxInk

```bash
pip install git+https://github.com/andata50/VoxInk.git
```

Or clone and install locally:

```bash
git clone https://github.com/andata50/VoxInk.git
cd VoxInk
pip install .
```

## Usage

### Basic usage

```bash
voxink song.mp3
```

This will:
1. Separate vocals from background music
2. Transcribe the vocals
3. Save a `.lrc` file next to the input

### Specify language

```bash
voxink song.mp3 -l de
```

### Choose output format

```bash
# LRC only (default)
voxink song.mp3 -f lrc

# SRT only
voxink song.mp3 -f srt

# Both
voxink song.mp3 -f both
```

### Use a larger model for better accuracy

```bash
voxink song.mp3 -m large
```

### Skip vocal separation

If your audio is already vocals-only:

```bash
voxink vocals.wav --skip-separation
```

### Only separate vocals

```bash
voxink song.mp3 --vocals-only
```

### Add metadata

```bash
voxink song.mp3 --title "Bis zum Schluss" --artist "Lacrimosa"
```

## Options

| Option | Description |
|--------|-------------|
| `-l, --language` | Language code (de, en, zh, ja...). Auto-detected if omitted |
| `-m, --model` | Whisper model: `tiny`, `base`, `small`, `medium` (default), `large` |
| `-f, --format` | Output: `lrc` (default), `srt`, `both` |
| `-o, --output` | Custom output path |
| `--skip-separation` | Skip Demucs, transcribe directly |
| `--vocals-only` | Only separate vocals |
| `--title` | Song title for LRC metadata |
| `--artist` | Artist name for LRC metadata |

## Model size guide

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 39 MB | Fastest | Low |
| base | 74 MB | Fast | Fair |
| small | 244 MB | Medium | Good |
| medium | 769 MB | Slow | Great |
| large | 1550 MB | Slowest | Best |

## License

MIT
