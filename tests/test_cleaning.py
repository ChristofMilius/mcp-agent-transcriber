"""tests/test_cleaning.py — caption/VTT → agent-friendly text."""
from __future__ import annotations

from mcp_agent_transcriber.cleaning import (
    parse_vtt_cues,
    render_vtt,
    strip_inline_tags,
    text_from_segments,
    text_from_vtt,
)

ROLLING_VTT = """WEBVTT
Kind: captions
Language: en

1
00:00:00.000 --> 00:00:01.000
<00:00:00.000><c>hello</c><00:00:00.300><c> world</c>

2
00:00:01.000 --> 00:00:02.000
hello world

3
00:00:02.000 --> 00:00:03.000
hello world <00:00:02.400><c>test</c><00:00:02.700><c> transcript</c>

4
00:00:03.000 --> 00:00:04.000
test transcript
"""


class TestVttParsing:
    def test_parse_vtt_cues_extracts_segments(self):
        cues = parse_vtt_cues(ROLLING_VTT)
        assert len(cues) == 4
        assert cues[0]["start"] == 0.0
        assert cues[0]["end"] == 1.0
        assert "world" in cues[0]["text"]

    def test_text_from_vtt_dedupes_rolling_captions(self):
        text = text_from_vtt(ROLLING_VTT)
        assert text == "hello world test transcript"

    def test_strip_inline_tags(self):
        assert strip_inline_tags("<00:00:01.234><c>hi</c>") == "hi"

    def test_whitespace_collapsed_across_lines(self):
        text = text_from_vtt("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n  hello   world \n\nx")
        assert text == "hello world"


class TestSegments:
    def test_text_from_segments_dedupes_consecutive(self):
        segs = [
            {"start": 0.0, "text": "one two"},
            {"start": 1.0, "text": "two three"},
            {"start": 2.0, "text": "two three"},
            {"start": 3.0, "text": "four"},
            {"start": 4.0, "text": ""},
        ]
        assert text_from_segments(segs) == "one two two three four"

    def test_render_vtt_uses_duration_when_end_missing(self):
        segs = [{"start": 0.0, "duration": 1.5, "text": "hi"}]
        vtt = render_vtt(segs)
        assert "00:00:00.000 --> 00:00:01.500" in vtt
        assert "WEBVTT" in vtt

    def test_render_vtt_uses_end_when_present(self):
        segs = [{"start": 2.0, "end": 3.5, "text": "bye"}]
        assert "00:00:02.000 --> 00:00:03.500" in render_vtt(segs)


class TestRoundTrip:
    def test_render_then_parse_is_lossless_for_text(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "line one"},
            {"start": 1.0, "end": 2.0, "text": "line two"},
        ]
        vtt = render_vtt(segs)
        assert text_from_vtt(vtt) == "line one line two"
