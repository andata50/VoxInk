"""VoxInk Web Interface - Gradio app for Hugging Face Spaces."""

import os
import tempfile
from pathlib import Path

import gradio as gr

# Detect if running on HF Spaces with ZeroGPU
IS_SPACES = os.environ.get("SPACE_ID") is not None
if IS_SPACES:
    import spaces

from voxink.ffmpeg_utils import ensure_ffmpeg_in_path
from voxink.separator import separate_vocals
from voxink.transcriber import transcribe
from voxink.aligner import align_lyrics
from voxink.converter import segments_to_lrc, segments_to_srt, save_lrc, save_srt


def _get_device():
    """Return 'cuda' if available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _process_audio_inner(
    audio_path: str,
    lyrics_text: str,
    language: str,
    model_size: str,
    output_format: str,
    skip_separation: bool,
    word_level: bool,
    title: str,
    artist: str,
) -> tuple[str, str | None]:
    """Core processing logic."""
    if audio_path is None:
        return "Please upload an audio file.", None

    audio_path = Path(audio_path)
    lang = language if language != "Auto" else None
    has_lyrics = lyrics_text and lyrics_text.strip()
    device = _get_device()

    # Step 1: Vocal separation
    if skip_separation:
        vocals_path = audio_path
    else:
        vocals_path = separate_vocals(str(audio_path), device=device)

    # Step 2: Transcribe or Align
    if has_lyrics:
        segments = align_lyrics(str(vocals_path), lyrics_text, language=lang, model_size=model_size, device=device)
    else:
        segments = transcribe(str(vocals_path), language=lang, model_size=model_size, device=device)

    if not segments:
        return "No lyrics detected in the audio.", None

    # Step 3: Generate output
    metadata = {}
    if title:
        metadata["title"] = title
    if artist:
        metadata["artist"] = artist

    if output_format in ("LRC", "Both"):
        preview = segments_to_lrc(segments, metadata or None, word_level=word_level)
    else:
        preview = segments_to_srt(segments)

    output_dir = Path(tempfile.mkdtemp())
    stem = audio_path.stem

    if output_format in ("LRC", "Both"):
        lrc_path = output_dir / f"{stem}.lrc"
        save_lrc(segments, str(lrc_path), metadata or None, word_level=word_level)

    if output_format in ("SRT", "Both"):
        srt_path = output_dir / f"{stem}.srt"
        save_srt(segments, str(srt_path))

    if output_format == "SRT":
        download_path = str(srt_path)
    else:
        download_path = str(lrc_path)

    return preview, download_path


# Apply @spaces.GPU decorator only on HF Spaces
if IS_SPACES:
    @spaces.GPU
    def process_audio(audio_path, lyrics_text, language, model_size, output_format,
                      skip_separation, word_level, title, artist):
        return _process_audio_inner(audio_path, lyrics_text, language, model_size,
                                    output_format, skip_separation, word_level, title, artist)
else:
    def process_audio(audio_path, lyrics_text, language, model_size, output_format,
                      skip_separation, word_level, title, artist):
        return _process_audio_inner(audio_path, lyrics_text, language, model_size,
                                    output_format, skip_separation, word_level, title, artist)


LANGUAGES = [
    "Auto", "de", "en", "zh", "ja", "ko", "fr", "es", "it", "pt",
    "ru", "nl", "pl", "sv", "tr", "ar", "hi", "th", "vi",
]

with gr.Blocks(title="VoxInk", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # VoxInk
        **Extract timed lyrics from songs automatically.**

        Upload a song → get synced lyrics (LRC/SRT).

        Two modes:
        - **Auto transcribe**: leave lyrics blank, VoxInk transcribes and times everything
        - **Align mode**: paste your own lyrics, VoxInk only adds timestamps
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="Upload Audio")
            lyrics_input = gr.Textbox(
                label="Lyrics Text (optional - paste to use align mode)",
                placeholder="Paste lyrics here, one line per line...\n\nLeave blank to auto-transcribe.",
                lines=8,
            )
            language = gr.Dropdown(choices=LANGUAGES, value="Auto", label="Language")
            model_size = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large"],
                value="small",
                label="Whisper Model (larger = more accurate but slower)",
            )
            output_format = gr.Dropdown(
                choices=["LRC", "SRT", "Both"],
                value="LRC",
                label="Output Format",
            )
            skip_separation = gr.Checkbox(
                value=False,
                label="Skip vocal separation (if audio is already vocals-only)",
            )
            word_level = gr.Checkbox(
                value=False,
                label="Word-level timestamps (per-word timing in LRC)",
            )

            with gr.Row():
                title = gr.Textbox(label="Song Title (optional)", placeholder="e.g. Bis zum Schluss")
                artist = gr.Textbox(label="Artist (optional)", placeholder="e.g. Lacrimosa")

            submit_btn = gr.Button("Generate Lyrics", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Lyrics Preview",
                lines=20,
                show_copy_button=True,
            )
            output_file = gr.File(label="Download Lyrics File")

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, lyrics_input, language, model_size, output_format, skip_separation, word_level, title, artist],
        outputs=[output_text, output_file],
    )

    gr.Markdown(
        """
        ---
        **Tips:**
        - **Align mode** gives more accurate lyrics text — paste your own lyrics and let VoxInk handle the timing
        - For best results, use `medium` or `large` model
        - Specify the language if auto-detection is inaccurate
        - Skip vocal separation if your audio is already isolated vocals
        """
    )

if __name__ == "__main__":
    ensure_ffmpeg_in_path()
    demo.launch()
