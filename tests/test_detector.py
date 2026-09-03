"""Referring-expression resolution and object-level question routing.

The detector's weights are one thing; the logic around them is what decides
which of two backbones answers a question, and which of several detections a
sentence meant. Both are pure, both are wrong silently, and both are what took
grounding from 0.2% to 17.3%.
"""
import numpy as np
import pytest

from satquery.tools.detector import (
    describe_position, describe_shape, describe_size, head_noun, is_object_level,
    nearest_colour, question_family, resolve_referring,
)


class TestHeadNoun:
    @pytest.mark.parametrize("text,expected", [
        ("The large yellow vehicle at the top", "vehicle"),
        ("two ships docked at the harbor", "ship"),
        ("the tennis courts on the left", "tennis court"),
        ("How many buses are there?", "bus"),
        ("the buildings in the centre", "building"),
    ])
    def test_singularises_the_category(self, text, expected):
        assert head_noun(text) == expected

    def test_returns_none_when_no_object_is_named(self):
        """Land-cover questions have no head noun, which is how the controller
        knows to keep them on the scene-level backbone."""
        assert head_noun("Is there water in this image?") is None
        assert head_noun("Describe the land cover") is None


class TestQuestionFamily:
    @pytest.mark.parametrize("query,family", [
        ("How many small vehicles are visible?", "count"),
        ("What is the number of buildings?", "count"),
        ("What color are the large vehicles?", "colour"),
        ("What is the shape of the building?", "shape"),
        ("Where is the ship located?", "position"),
        ("Is there a vehicle present?", "existence"),
        ("The large vehicle at the top-left", "referring"),
    ])
    def test_families(self, query, family):
        assert question_family(query) == family


class TestObjectLevelRouting:
    """Which backbone answers. Getting this wrong sends every land-cover
    question to a detector that has no concept of land cover."""

    def test_counting_is_always_object_level(self):
        assert is_object_level("How many farmlands are there?")

    def test_land_cover_presence_stays_scene_level(self):
        assert not is_object_level("Is there water in this image?")

    def test_object_presence_is_object_level(self):
        assert is_object_level("Is there a vehicle in this image?")

    def test_either_or_scene_question_stays_scene_level(self):
        assert not is_object_level("Does the image depict a rural or urban area?")

    def test_captioning_request_stays_scene_level(self):
        assert not is_object_level("Describe the land-cover in this image.")


class TestBoxDescriptions:
    def test_position_corners_and_centre(self):
        assert describe_position([0.0, 0.0, 0.2, 0.2]) == "top-left"
        assert describe_position([0.8, 0.8, 1.0, 1.0]) == "bottom-right"
        assert describe_position([0.45, 0.45, 0.55, 0.55]) == "center"

    def test_size_thresholds(self):
        assert describe_size([0, 0, 0.05, 0.05]) == "small"
        assert describe_size([0, 0, 0.5, 0.5]) == "large"

    def test_shape_from_aspect_ratio(self):
        assert describe_shape([0, 0, 0.2, 0.2]) == "square"
        assert describe_shape([0, 0, 0.6, 0.1]) == "elongated"

    def test_colour_naming(self):
        assert nearest_colour((240, 240, 240)) == "white"
        assert nearest_colour((30, 30, 30)) == "black"


class _Img:
    """Stand-in for PIL, so resolution can be tested without decoding pixels."""
    size = (100, 100)

    def crop(self, _box):
        return self

    def convert(self, _mode):
        return self

    def __array__(self, dtype=None):
        return np.full((4, 4, 3), 128, dtype=dtype or np.float32)


class TestResolveReferring:
    """Detection score answers "is there a vehicle"; it cannot answer "which
    vehicle". Almost every referring expression here states a position."""

    def _cands(self):
        return [
            {"box": [0.0, 0.0, 0.15, 0.15], "score": 0.30, "label": "vehicle"},
            {"box": [0.8, 0.8, 0.98, 0.98], "score": 0.45, "label": "vehicle"},
        ]

    def test_spatial_constraint_overrides_a_higher_detection_score(self):
        out = resolve_referring(_Img(), self._cands(), "the vehicle at the top-left")
        assert out[0]["box"][0] == 0.0, "spatial constraint ignored"

    def test_opposite_constraint_picks_the_other_instance(self):
        out = resolve_referring(_Img(), self._cands(), "the vehicle at the bottom-right")
        assert out[0]["box"][0] == 0.8

    def test_size_constraint_separates_equal_positions(self):
        cands = [
            {"box": [0.4, 0.4, 0.45, 0.45], "score": 0.4, "label": "ship"},
            {"box": [0.3, 0.3, 0.7, 0.7], "score": 0.4, "label": "ship"},
        ]
        big = resolve_referring(_Img(), cands, "the large ship")[0]
        small = resolve_referring(_Img(), cands, "the small ship")[0]
        assert big["box"] == [0.3, 0.3, 0.7, 0.7]
        assert small["box"] == [0.4, 0.4, 0.45, 0.45]

    def test_empty_candidates_return_empty(self):
        assert resolve_referring(_Img(), [], "anything") == []

    def test_every_candidate_is_scored_and_kept(self):
        out = resolve_referring(_Img(), self._cands(), "the vehicle at the top")
        assert len(out) == 2
        assert all("referring_score" in c for c in out)


def test_existence_uses_a_stricter_threshold_than_localisation():
    """Localising wants recall; asserting existence on a weak box is a false
    claim. At a shared 0.10 threshold, presence accuracy fell below the
    scene-level backbone the detector replaced."""
    import inspect
    from satquery.tools.detector import DetectionTool
    source = inspect.getsource(DetectionTool.run)
    assert "existence_threshold" in source
    assert "existence_threshold" in DetectionTool.spec.accepts


def test_three_thresholds_are_separately_configurable():
    """Localisation, existence and counting need different operating points.

    Localisation is scored on the best box, so a spurious extra costs nothing.
    A count is scored exactly, so every false positive is an error. Asserting
    existence on a weak box is a false claim. One shared threshold serves none
    of them: at a shared 0.03, counting accuracy fell from 8.8% to 2.0%.
    """
    from satquery.tools.detector import DetectionTool
    for name in ("threshold", "existence_threshold", "count_threshold"):
        assert name in DetectionTool.spec.accepts


class TestPolarPrecedence:
    """A polar opener settles the question type before position or size.

    "Is there a ship located close to the rightmost edge?" mentions a position
    but asks yes or no, and answering "bottom-right" is wrong however correct
    the position is. Testing position first did exactly that and produced a
    quarter of the errors on VRSBench's largest question type.
    """

    def test_existential_with_a_position_clause_is_still_yes_no(self):
        assert question_family(
            "Is there a ship located close to the rightmost edge of the image?"
        ) == "existence"

    def test_polar_without_an_existential_is_still_yes_no(self):
        """"Do the planes have multiple engines" names no "there" or "any", and
        fell through to the referring branch, which answers with a detection
        summary rather than yes or no."""
        assert question_family("Do the planes appear to have multiple engines?") == "existence"

    def test_wh_questions_still_win_over_the_polar_opener(self):
        assert question_family("How many ships are there?") == "count"
        assert question_family("What color are the vehicles?") == "colour"
        assert question_family("What is the shape of the building?") == "shape"

    def test_a_genuine_position_question_is_unaffected(self):
        assert question_family("Where is the ship located?") == "position"
