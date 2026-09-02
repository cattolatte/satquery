"""The answer-decoding logic in the evaluation harnesses.

These functions turn a tool's prose into the token a benchmark scores against.
They are pure, and when they are wrong they do not crash -- they silently
report the wrong accuracy, which is worse. Three of the bugs they guard were
found this way: counting scored 12.5% by scraping stray digits, change ratios
scored 0.0% from a format mismatch, and "non-change" questions were answered
with the un-inverted percentage.
"""
from eval.cdvqa import answer_from, ratio_bin, subject_of
from eval.rsvqa import predicted_answer, question_type
from satquery.schema import Evidence


class TestRSVQAQuestionTypes:
    def test_counting_is_recognised(self):
        assert question_type("How many farmlands are there?") == "count"
        assert question_type("What is the number of buildings?") == "count"

    def test_object_level_presence_is_separated_from_scene_level(self):
        assert question_type("Is a circular building present?") == "presence-object"
        assert question_type("Is there water?") == "presence-scene"

    def test_rural_urban_is_its_own_type(self):
        assert question_type("Is it a rural or an urban area") == "rural/urban"

    def test_comparison(self):
        assert question_type("Are there more farmlands than water areas?") == "comparison"


class TestRSVQADecoding:
    def test_counting_is_never_guessed(self):
        """Stray digits in the prose must not be read as a count."""
        text = "Yes — 'urban fabric' ranks 19 of 21 (similarity 0.211)."
        assert predicted_answer(text, "count") == "unsupported"

    def test_yes_and_no_are_read_from_the_leading_token(self):
        assert predicted_answer("Yes — 'water' ranks 2 of 21.", "presence-scene") == "yes"
        assert predicted_answer("No — 'water' ranks 19 of 21.", "presence-scene") == "no"

    def test_either_or_answer_is_read_directly(self):
        assert predicted_answer("Urban — scored urban 0.28 vs rural 0.21.",
                                "rural/urban") == "urban"
        assert predicted_answer("Rural — scored rural 0.29 vs urban 0.20.",
                                "rural/urban") == "rural"


class TestRatioBins:
    def test_exact_zero_is_distinct_from_the_first_bin(self):
        """CDVQA's gold answers carry a bare "0" as well as "0_to_10"."""
        assert ratio_bin(0.0) == "0"
        assert ratio_bin(0.05) == "0_to_10"

    def test_bins_are_ten_points_wide(self):
        assert ratio_bin(0.25) == "20_to_30"
        assert ratio_bin(0.91) == "90_to_100"

    def test_full_change_does_not_overflow_the_last_bin(self):
        assert ratio_bin(1.0) == "90_to_100"


class TestCDVQASubjects:
    def test_second_classes_are_recognised(self):
        assert subject_of("Did the regions of trees change?") == "trees"
        assert subject_of("Have the areas of water changed?") == "water"
        assert subject_of("Did the areas of non-vegetated ground surface change?") \
            == "non-vegetated ground surface"

    def test_low_vegetation_is_not_swallowed_by_a_broader_match(self):
        assert subject_of("Have the areas of low vegetation changed?") == "low vegetation"

    def test_unknown_subject_is_none(self):
        assert subject_of("What is the percentage of changed areas?") is None


class TestCDVQADecoding:
    def _ev(self, **areas):
        return [Evidence("area_delta", d, c, abs(d)) for c, d in areas.items()]

    def test_direction_reads_the_sign_of_the_area_delta(self):
        ev = [Evidence("area_delta", 0.3, "buildings", 0.3)]
        assert answer_from(ev, "", "increase_or_not", "buildings") == "yes"
        assert answer_from(ev, "", "decrease_or_not", "buildings") == "no"

    def test_change_below_threshold_is_no(self):
        ev = [Evidence("area_delta", 0.01, "water", 0.01)]
        assert answer_from(ev, "", "change_or_not", "water") == "no"

    def test_change_above_threshold_is_yes(self):
        ev = [Evidence("area_delta", -0.4, "water", 0.4)]
        assert answer_from(ev, "", "change_or_not", "water") == "yes"

    def test_change_to_what_excludes_the_subject_and_picks_the_gainer(self):
        ev = [Evidence("area_delta", -0.5, "water", 0.5),
              Evidence("area_delta", 0.3, "buildings", 0.3),
              Evidence("area_delta", 0.1, "trees", 0.1)]
        assert answer_from(ev, "", "change_to_what", "water") == "buildings"

    def test_non_change_question_is_inverted(self):
        ev = []
        text = "30% of the scene changed."
        assert answer_from(ev, text, "change_ratio", None,
                           "What is the percentage of non-change regions?") == "70_to_80"
        assert answer_from(ev, text, "change_ratio", None,
                           "What is the percentage of changed areas?") == "30_to_40"

    def test_per_class_ratio_uses_the_subject_not_the_scene(self):
        ev = [Evidence("area_delta", 0.05, "buildings", 0.05)]
        text = "90% of the scene changed."
        assert answer_from(ev, text, "change_ratio_types", "buildings",
                           "What is the change proportion of buildings?") == "0_to_10"
