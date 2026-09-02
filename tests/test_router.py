"""Query routing and input reconciliation.

The representative queries in the problem statement are the specification, so
they are the test cases. The reconciliation tests cover the harder half: what
happens when the query asks for something the supplied images cannot support.
"""
from satquery.controller.router import classify, reconcile
from satquery.schema import InputKind, Task


class TestClassify:
    """The problem statement's own example queries."""

    def test_change_description(self):
        assert classify(
            "Describe the changes between these two images"
        ) is Task.CHANGE_DESCRIPTION

    def test_change_vqa(self):
        assert classify(
            "Has new construction appeared between these two dates?"
        ) is Task.CHANGE_VQA

    def test_grounding(self):
        assert classify(
            "Show me the location of the airport runway in this image"
        ) is Task.GROUNDING

    def test_caption(self):
        assert classify("Describe this satellite image") is Task.CAPTION

    def test_vqa(self):
        assert classify("Is there water in this image?") is Task.VQA

    def test_cross_modal(self):
        assert classify(
            "Compare this SAR image with the optical image"
        ) is Task.CROSS_MODAL


class TestReconcile:
    """Reconciliation must degrade explicitly, never silently."""

    def test_pair_task_on_single_image_downgrades_with_a_note(self):
        task, notes = reconcile(Task.CHANGE_DESCRIPTION, InputKind.SINGLE)
        assert task is Task.VQA
        assert notes, "a downgrade must be recorded in the trace"
        assert "two images" in notes[0]

    def test_supported_task_passes_through_without_notes(self):
        task, notes = reconcile(Task.VQA, InputKind.SINGLE)
        assert task is Task.VQA
        assert notes == []

    def test_invalid_input_supports_nothing(self):
        task, notes = reconcile(Task.VQA, InputKind.INVALID)
        assert task is Task.UNSUPPORTED
        assert notes

    def test_cross_modal_asked_of_two_dates_falls_back_to_change(self):
        task, notes = reconcile(Task.CROSS_MODAL, InputKind.BI_TEMPORAL_PAIR)
        assert task is Task.CHANGE_DESCRIPTION
        assert notes

    def test_change_asked_of_a_cross_modal_pair_falls_back(self):
        task, notes = reconcile(Task.CHANGE_DESCRIPTION, InputKind.CROSS_MODAL_PAIR)
        assert task is Task.CROSS_MODAL
        assert notes

    def test_every_reconciled_task_is_actually_permitted(self):
        """Reconciliation must never return a task the inputs cannot support."""
        from satquery.controller.router import _ALLOWED
        for kind, allowed in _ALLOWED.items():
            for task in Task:
                out, _ = reconcile(task, kind)
                if out is Task.UNSUPPORTED:
                    continue
                assert out in allowed, f"{kind.value}+{task.value} -> illegal {out.value}"


class TestChangeVerbAmbiguity:
    """Change verbs whose reading depends on grammatical aspect.

    "are grown" is passive agriculture; "has grown" is change over time. The
    router has to tell them apart, or every crop question on a bi-temporal pair
    routes to the change tool.
    """

    def test_passive_grown_is_plain_vqa(self):
        assert classify("What crops are grown here?") is Task.VQA

    def test_perfect_grown_is_change(self):
        assert classify("Has the city grown?") is Task.CHANGE_VQA

    def test_same_without_perfect_aspect_is_vqa(self):
        assert classify("Are these fields the same crop?") is Task.VQA

    def test_unchanged_is_change_regardless_of_aspect(self):
        assert classify("Is the forest area unchanged?") is Task.CHANGE_VQA

    def test_counting_question_stays_vqa(self):
        assert classify("How many buildings are in this image?") is Task.VQA

    def test_since_a_year_is_temporal(self):
        assert classify("Did the reservoir shrink since 2019?") is Task.CHANGE_VQA


class TestProblemStatementQueries:
    """The five representative queries quoted verbatim in the problem statement.

    These are the specification, so a regression here is a conformance failure
    rather than a quality one.
    """

    def test_describe_land_cover_and_objects(self):
        assert classify(
            "Describe the land-cover and major objects visible in this image."
        ) is Task.CAPTION

    def test_highlight_the_water_body(self):
        assert classify(
            "Highlight the water body referred to in the query."
        ) is Task.GROUNDING

    def test_what_changed_and_where(self):
        """Contains "did ... change", which reads as a polar question unless
        the leading interrogative is given priority."""
        assert classify(
            "What changed between these two dates, and where did the change occur?"
        ) is Task.CHANGE_DESCRIPTION

    def test_use_optical_and_sar_together(self):
        assert classify(
            "Use the optical and SAR images together to identify built-up and "
            "water-covered regions."
        ) is Task.CROSS_MODAL

    def test_has_built_up_area_increased(self):
        assert classify(
            "Has the built-up area increased, decreased, or remained unchanged?"
        ) is Task.CHANGE_VQA

    def test_open_change_question_outranks_polar_reading(self):
        """A wh-question about change is a description request; a polar
        question about the same scene is change VQA."""
        assert classify("Did the forest area change between the two dates?") is Task.CHANGE_VQA
        assert classify("What changed between the two dates?") is Task.CHANGE_DESCRIPTION
