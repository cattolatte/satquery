"""Report rendering.

The report is a deliverable the statement asks for, and it is generated after
the answer already exists -- a crash here throws away work that succeeded. It
also embeds text the user typed, so escaping matters.
"""
import re
from pathlib import Path

import pytest

from satquery.report import _data_uri, build_html

IMAGE = next(Path("data/adapt/images").glob("*.png"), None)


def _payload(**over):
    base = {"text": "an answer", "confidence": 0.5, "evidence": [],
            "query": "a question", "trace": {"calls": []}}
    base.update(over)
    return base


class TestRobustness:
    def test_minimal_payload_renders(self):
        assert "<!doctype html>" in build_html(_payload(trace={}), []).lower()

    def test_missing_trace_key(self):
        payload = {"text": "x", "confidence": 0.0, "evidence": []}
        assert build_html(payload, [])

    def test_missing_image_is_skipped_not_fatal(self):
        assert build_html(_payload(), ["/does/not/exist.png"])

    def test_unreadable_image_yields_no_uri(self):
        assert _data_uri("/does/not/exist.png") is None

    def test_confidence_of_zero_renders(self):
        assert "0%" in build_html(_payload(confidence=0.0), [])


class TestEscaping:
    """Checked by looking for an executable tag, not for a substring: an
    escaped tag still contains its own text, so substring matching reports
    false positives."""

    HOSTILE = "<img src=x onerror=alert(1)>"

    def _rendered(self):
        return build_html(_payload(query=self.HOSTILE, text="<script>alert(1)</script>",
                                   trace={"calls": [], "rejected": ["<i>x</i>"],
                                          "task": "<b>vqa</b>"}), [])

    def test_no_executable_tag_survives(self):
        html = self._rendered()
        assert not re.findall(
            r"<\s*(script|img|iframe|svg|body)\b[^>]*(onerror|onload)", html, re.I)

    def test_hostile_text_is_present_but_escaped(self):
        html = self._rendered()
        assert "&lt;img src=x" in html
        assert self.HOSTILE not in html

    def test_script_tag_is_escaped(self):
        assert "<script>alert(1)</script>" not in self._rendered()


@pytest.mark.skipif(IMAGE is None, reason="no sample image available")
class TestSelfContained:
    def test_images_are_embedded_not_linked(self):
        html = build_html(_payload(), [str(IMAGE)])
        assert "data:image/png;base64," in html
        assert not re.search(r'(src|href)="(?!data:)https?://', html)

    def test_boxes_are_drawn_over_the_image(self):
        html = build_html(_payload(evidence=[
            {"kind": "bbox", "data": [0.1, 0.1, 0.5, 0.5], "label": "ship", "score": 0.9}
        ]), [str(IMAGE)])
        assert 'class="box"' in html and "ship" in html
