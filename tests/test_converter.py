"""Tests for the converter module."""

from voxink.converter import segments_to_lrc, segments_to_srt, _format_lrc_time, _format_srt_time


def test_format_lrc_time():
    assert _format_lrc_time(0) == "00:00.00"
    assert _format_lrc_time(65.5) == "01:05.50"
    assert _format_lrc_time(125.123) == "02:05.12"


def test_format_srt_time():
    assert _format_srt_time(0) == "00:00:00,000"
    assert _format_srt_time(65.5) == "00:01:05,500"
    assert _format_srt_time(3661.123) == "01:01:01,123"


def test_segments_to_lrc():
    segments = [
        {"start": 10.0, "end": 13.0, "text": "Hello world"},
        {"start": 15.5, "end": 18.0, "text": "Second line"},
    ]
    result = segments_to_lrc(segments)
    assert "[00:10.00]Hello world" in result
    assert "[00:15.50]Second line" in result


def test_segments_to_lrc_with_metadata():
    segments = [{"start": 0, "end": 3, "text": "Test"}]
    result = segments_to_lrc(segments, metadata={"title": "My Song", "artist": "Artist"})
    assert "[ti:My Song]" in result
    assert "[ar:Artist]" in result
    assert "[tool:VoxInk]" in result


def test_segments_to_srt():
    segments = [
        {"start": 10.0, "end": 13.5, "text": "Hello world"},
    ]
    result = segments_to_srt(segments)
    assert "1" in result
    assert "00:00:10,000 --> 00:00:13,500" in result
    assert "Hello world" in result
