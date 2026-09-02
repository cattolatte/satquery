"""
Shared types.

The execution trace is a first-class output, not a debug log. The problem
statement is explicit about this:

    "only the observable execution trace, including the selected task, models
    or tools, permitted parameters, and outputs will be evaluated. Internal
    reasoning text is neither required nor evaluated."

So the trace is a graded artefact. It is typed, validated, and serialisable,
and nothing that happens is allowed to go unrecorded in it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Task(str, Enum):
    """Tasks the controller can route to. Named after the problem statement's
    own vocabulary so the trace reads against the requirements directly."""

    VQA = "vqa"                       # mandatory single-image baseline
    CAPTION = "caption"               # single-image, one of two options
    GROUNDING = "grounding"           # single-image, the other option
    CHANGE_DESCRIPTION = "change_description"
    CHANGE_VQA = "change_vqa"
    CROSS_MODAL = "cross_modal"       # optical + SAR joint extraction
    UNSUPPORTED = "unsupported"


class Modality(str, Enum):
    OPTICAL = "optical"               # optical / multispectral
    SAR = "sar"
    UNKNOWN = "unknown"


class InputKind(str, Enum):
    """The three input configurations the problem statement defines."""

    SINGLE = "single"
    CROSS_MODAL_PAIR = "cross_modal_pair"
    BI_TEMPORAL_PAIR = "bi_temporal_pair"
    INVALID = "invalid"


@dataclass
class ImageMeta:
    """What we know about one input image before any model sees it."""

    path: str
    fmt: str                          # GeoTIFF, TIFF, PNG, JPEG
    width: int
    height: int
    bands: int
    modality: Modality = Modality.UNKNOWN
    georeferenced: bool = False
    crs: str | None = None
    bounds: tuple[float, float, float, float] | None = None
    acquired: str | None = None       # ISO date if present in metadata
    notes: list[str] = field(default_factory=list)


@dataclass
class ToolCall:
    """One executed step. Every field here is evaluated, so none is optional
    in spirit even where it is in code."""

    tool: str
    task: Task
    params: dict[str, Any]
    ms: float
    ok: bool
    error: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    """Visual evidence returned alongside the text answer."""

    kind: str                         # bbox | mask | change_map | overlay
    data: Any
    label: str = ""
    score: float | None = None


@dataclass
class Trace:
    """The auditable execution summary. This is the graded artefact."""

    query: str
    input_kind: InputKind
    images: list[ImageMeta]
    task: Task
    calls: list[ToolCall] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)   # validation failures
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    """What the user gets back."""

    text: str
    confidence: float
    evidence: list[Evidence] = field(default_factory=list)
    trace: Trace | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"text": self.text, "confidence": self.confidence,
             "evidence": [asdict(e) for e in self.evidence]}
        if self.trace:
            d["trace"] = self.trace.to_dict()
        return d
