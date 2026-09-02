"""The pure logic inside the specialist tools.

These run without torch or any weights: the parts tested here are the ones that
decide what a question is about and how much to trust an answer, and they are
wrong in ways no model quality can compensate for.
"""
import numpy as np
import pytest

from satquery.tools.specialists import (
    LAND_COVER, _grounding_confidence, _referring_phrase, _resolve_target,
)


class TestReferringPhrase:
    """Grounding feeds this straight to the text encoder, so filler words in
    the phrase directly weaken the heat map."""

    def test_strips_imperative_and_trailing_scene_reference(self):
        assert _referring_phrase(
            "Show me the location of the forest in this image") == "forest"

    def test_strips_bare_imperative(self):
        assert _referring_phrase("Highlight the airport runway") == "airport runway"

    def test_honours_explicit_ref_tags(self):
        assert _referring_phrase(
            "locate <ref>the western reservoir</ref> please") == "the western reservoir"

    def test_never_returns_empty(self):
        for q in ("find", "where is", "show me"):
            assert _referring_phrase(q).strip()


class TestResolveTarget:
    """Everyday words have to reach CORINE class names that do not contain them."""

    @pytest.mark.parametrize("word,expected", [
        ("water", "inland waters"),
        ("forest", "broad-leaved forest"),
        ("roads", "road network"),
        ("farmland", "arable land"),
        ("runway", "airport runways"),
        ("beach", "beaches dunes sands"),
    ])
    def test_synonyms_reach_their_classes(self, word, expected):
        got = _resolve_target(f"Is there {word} in this image?", LAND_COVER)
        assert expected in got, f"{word!r} -> {got}"

    def test_verbatim_class_name_matches(self):
        assert "mixed forest" in _resolve_target(
            "Is there mixed forest here?", LAND_COVER)

    def test_forest_resolves_to_every_forest_class(self):
        got = _resolve_target("Is there forest in this image?", LAND_COVER)
        for cls in ("broad-leaved forest", "coniferous forest", "mixed forest"):
            assert cls in got

    def test_unrelated_question_resolves_to_nothing(self):
        assert _resolve_target("What is the weather like?", LAND_COVER) == []

    def test_result_has_no_duplicates(self):
        got = _resolve_target("Is there water, lake and river here?", LAND_COVER)
        assert len(got) == len(set(got))


class TestGroundingConfidence:
    """The bug this guards: min-max normalising the heat map makes its maximum
    1.0 by construction, so every localisation reported perfect confidence."""

    def test_a_sharp_peak_is_confident(self):
        raw = np.zeros((7, 7)); raw[3, 3] = 5.0
        assert _grounding_confidence(raw, raw > 1.0) > 0.6

    def test_flat_similarity_is_not_confident(self):
        raw = np.full((7, 7), 0.3)
        sel = np.zeros((7, 7), dtype=bool); sel[0, 0] = True
        assert _grounding_confidence(raw, sel) < 0.3

    def test_selecting_almost_everything_is_not_a_localisation(self):
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(7, 7))
        sel = np.ones((7, 7), dtype=bool); sel[0, 0] = False
        assert _grounding_confidence(raw, sel) < 0.25

    def test_degenerate_masks_do_not_crash(self):
        raw = np.random.default_rng(1).normal(size=(7, 7))
        for sel in (np.zeros((7, 7), dtype=bool), np.ones((7, 7), dtype=bool)):
            c = _grounding_confidence(raw, sel)
            assert 0.0 <= c <= 1.0

    def test_confidence_is_never_pinned_at_one(self):
        """A realistic map must not saturate the way the old version did."""
        rng = np.random.default_rng(2)
        for _ in range(25):
            raw = rng.normal(size=(7, 7))
            thr = raw.min() + 0.6 * np.ptp(raw)
            c = _grounding_confidence(raw, raw >= thr)
            assert c < 1.0


def test_referring_phrase_on_the_statements_own_query():
    """'Highlight the water body referred to in the query.' is quoted verbatim
    in the problem statement; the trailing clause is not part of the subject."""
    assert _referring_phrase(
        "Highlight the water body referred to in the query.") == "water body"
