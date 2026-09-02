"""
Query -> task classification, and whether that task is possible with the inputs.

The problem statement asks the controller to "interpret the query and classify
the requested task" and, separately, to "check the number, modality, format,
metadata, and compatibility of the input images". Those two must agree: a
change question over a single image is a well-formed query and an impossible
task, and saying so is more useful than answering it badly.

Routing is rule-based. That is a deliberate choice, not a placeholder:

  - It is auditable. The trace must record why a task was chosen, and a regex
    that matched is a better answer than "the model felt so".
  - It is instant, so the latency budget goes to the specialist models.
  - The vocabulary is small and closed. There are six tasks and the queries
    that select them use a narrow set of words.

`classify_llm` exists for queries the rules cannot place.
"""
from __future__ import annotations

import re

from ..schema import InputKind, Task

# Ordered: the first match wins, so the more specific patterns come first.
# Change and cross-modal intents are checked before the generic single-image
# ones, because "what changed in this built-up area" is a change question that
# also mentions land cover.
_PATTERNS: list[tuple[Task, re.Pattern[str]]] = [
    # Open questions about change ask for a description; polar ones ask for a
    # yes/no. "What changed between these two dates, and where did the change
    # occur?" contains "did ... change" and would otherwise be read as change
    # VQA, so a leading auxiliary disqualifies this pattern outright.
    (Task.CHANGE_DESCRIPTION, re.compile(
        r"^(?!\s*(has|have|had|did|does|do|is|are|was|were)\b).*?(?:"
        r"\b(what|which|where|how)\b.*\bchang(e|ed|es|ing)\b|"
        r"\bchang(e|ed|es|ing)\b.*\bbetween\b|\bcompare\b.*\bdates?\b|"
        r"\bbefore\b.*\bafter\b)", re.I)),
    # Change questions split by how ambiguous their verb is.
    #
    # Unambiguous verbs ("shrunk", "appeared") read as change after any polar
    # opener. Ambiguous ones do not: "What crops are grown here?" is a plain
    # VQA question, while "Has the city grown?" is a change question -- the
    # difference is perfect/past aspect, so those verbs require it.
    #
    # An explicitly temporal clause is sufficient on its own, because the
    # statement's own example ("Has new construction appeared between these
    # two dates?") carries no change verb at all.
    (Task.CHANGE_VQA, re.compile(
        r"\b(has|have|had|did|does|do|is|are|was|were)\b.*?("
        r"\b(increase[sd]?|decreas(e|ed|es)|shrunk|shrank|expand(ed|s)?|"
        r"reduc(e|ed|es)|chang(e|ed|es)|unchanged|appear(ed|s)?|"
        r"disappear(ed|s)?|emerg(e|ed|es)|vanish(ed|es)?|"
        r"construct(ed|ion)|demolish(ed)?)\b"
        r"|\bbetween\b.*\b(dates?|images?|acquisitions?|times?|years?)\b"
        r"|\bsince\b\s+\d|\bcompared\s+to\b)", re.I)),
    (Task.CHANGE_VQA, re.compile(
        r"\b(has|have|had|did)\b.*?\b(grown|grew|new|added|removed|built|"
        r"clear(ed)?|lost|gained|same)\b", re.I)),
    (Task.CROSS_MODAL, re.compile(
        r"\b(optical|multispectral)\b.*\b(sar|radar)\b|"
        r"\b(sar|radar)\b.*\b(optical|multispectral)\b|"
        r"\bboth\b.*\b(image|sensor|modalit)", re.I)),
    (Task.GROUNDING, re.compile(
        r"\b(highlight|locate|find|show me|point to|mark|where is|where are|"
        r"outline|segment)\b", re.I)),
    (Task.CAPTION, re.compile(
        r"\b(describe|caption|summari[sz]e|what does .* show|overview|"
        r"tell me about)\b", re.I)),
]

# Which tasks each input configuration can actually support.
_ALLOWED: dict[InputKind, set[Task]] = {
    InputKind.SINGLE: {Task.VQA, Task.CAPTION, Task.GROUNDING},
    InputKind.CROSS_MODAL_PAIR: {Task.CROSS_MODAL, Task.VQA, Task.CAPTION, Task.GROUNDING},
    InputKind.BI_TEMPORAL_PAIR: {Task.CHANGE_DESCRIPTION, Task.CHANGE_VQA,
                                 Task.VQA, Task.CAPTION, Task.GROUNDING},
    InputKind.INVALID: set(),
}

_NEEDS_PAIR = {Task.CHANGE_DESCRIPTION, Task.CHANGE_VQA, Task.CROSS_MODAL}


def classify(query: str) -> Task:
    """Best task for this query, ignoring what images are available.

    VQA is the fallback because the problem statement makes it the mandatory
    baseline: any question that is not clearly something else is a question
    about the image.
    """
    q = (query or "").strip()
    if not q:
        return Task.UNSUPPORTED
    for task, pattern in _PATTERNS:
        if pattern.search(q):
            return task
    return Task.VQA


def reconcile(task: Task, kind: InputKind) -> tuple[Task, list[str]]:
    """Reconcile the requested task with what the inputs can support.

    Returns the task to run and any notes for the trace. A change question
    asked of a single image is downgraded to VQA with the reason recorded,
    rather than either failing outright or silently pretending.
    """
    notes: list[str] = []
    allowed = _ALLOWED.get(kind, set())

    if not allowed:
        return Task.UNSUPPORTED, [f"input configuration {kind.value} supports no tasks"]

    if task in allowed:
        return task, notes

    if task in _NEEDS_PAIR and kind is InputKind.SINGLE:
        notes.append(
            f"query asks for {task.value}, which needs two images; one was supplied. "
            "Falling back to single-image VQA.")
        return Task.VQA, notes

    if task is Task.CROSS_MODAL and kind is InputKind.BI_TEMPORAL_PAIR:
        notes.append(
            "query asks for optical-SAR analysis, but both images share a modality. "
            "Falling back to change description.")
        return Task.CHANGE_DESCRIPTION, notes

    if task in {Task.CHANGE_DESCRIPTION, Task.CHANGE_VQA} and kind is InputKind.CROSS_MODAL_PAIR:
        notes.append(
            "query asks about change, but the pair is optical+SAR rather than two dates. "
            "Falling back to cross-modal analysis.")
        return Task.CROSS_MODAL, notes

    notes.append(f"{task.value} is not supported for {kind.value}; falling back to VQA")
    return Task.VQA, notes
