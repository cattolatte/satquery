"""
The tool registry.

The problem statement calls for the controller to "select one or more models or
tools from a predefined registry" and to "configure only permitted task
parameters". Both halves matter, and the second is the one that is easy to skip:

    "configure only PERMITTED task parameters"

So a tool declares which parameters it accepts, and the controller cannot pass
anything else. Unknown parameters are dropped and recorded in the trace rather
than forwarded, because a silently-accepted parameter is how a controller ends
up doing something no one specified.

Every tool is required to degrade rather than raise. A specialist model that is
unavailable should cost coverage, never the whole query.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..schema import Evidence, ImageMeta, Task, ToolCall


@dataclass
class ToolSpec:
    """What a tool is, declared rather than inferred."""

    name: str
    tasks: set[Task]
    accepts: set[str]                       # permitted parameter names
    needs_images: int = 1
    description: str = ""
    requires: list[str] = field(default_factory=list)   # python packages / weights


class Tool(ABC):
    """A specialist model or algorithm the controller can dispatch to."""

    spec: ToolSpec

    @abstractmethod
    def run(self, images: list[ImageMeta], query: str,
            params: dict[str, Any]) -> tuple[str, list[Evidence], float]:
        """Return (text, evidence, confidence). Must not raise for ordinary
        failures — return a low confidence and say so in the text."""

    def available(self) -> tuple[bool, str]:
        """Can this tool run right now? Checked before dispatch so the trace
        records the reason rather than an exception."""
        import importlib
        for mod in self.spec.requires:
            if importlib.util.find_spec(mod) is None:
                return False, f"missing dependency: {mod}"
        return True, ""

    def filter_params(self, params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Keep only permitted parameters. Returns (kept, dropped_names)."""
        kept = {k: v for k, v in params.items() if k in self.spec.accepts}
        dropped = sorted(set(params) - set(kept))
        return kept, dropped

    def invoke(self, images: list[ImageMeta], query: str,
               params: dict[str, Any]) -> tuple[ToolCall, str, list[Evidence], float]:
        """Run with timing, parameter filtering and failure capture, producing
        the ToolCall record the trace requires."""
        kept, dropped = self.filter_params(params)
        ok_dep, why = self.available()
        t0 = time.perf_counter()

        if not ok_dep:
            call = ToolCall(self.spec.name, next(iter(self.spec.tasks)), kept,
                            round((time.perf_counter() - t0) * 1000, 1), False, why)
            return call, f"[{self.spec.name} unavailable: {why}]", [], 0.0

        try:
            text, evidence, conf = self.run(images, query, kept)
            err = None
            ok = True
        except Exception as e:                                # noqa: BLE001
            text, evidence, conf, err, ok = f"[{self.spec.name} failed: {e}]", [], 0.0, str(e), False

        call = ToolCall(
            tool=self.spec.name,
            task=next(iter(self.spec.tasks)),
            params=kept,
            ms=round((time.perf_counter() - t0) * 1000, 1),
            ok=ok,
            error=err,
            outputs={"chars": len(text), "evidence": len(evidence),
                     "confidence": round(conf, 3)},
        )
        if dropped:
            call.outputs["dropped_params"] = dropped
        return call, text, evidence, conf


class Registry:
    """The predefined registry the problem statement asks for."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.spec.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.spec.name}")
        self._tools[tool.spec.name] = tool
        return tool

    def for_task(self, task: Task) -> list[Tool]:
        """Tools that can serve this task, available ones first so the
        controller prefers a working specialist over a fallback."""
        candidates = [t for t in self._tools.values() if task in t.spec.tasks]
        return sorted(candidates, key=lambda t: (not t.available()[0], t.spec.name))

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def describe(self) -> list[dict[str, Any]]:
        """Registry contents, for the UI and the trace."""
        out = []
        for t in self._tools.values():
            ok, why = t.available()
            out.append({"name": t.spec.name,
                        "tasks": sorted(x.value for x in t.spec.tasks),
                        "accepts": sorted(t.spec.accepts),
                        "needs_images": t.spec.needs_images,
                        "available": ok, "reason": why,
                        "description": t.spec.description})
        return sorted(out, key=lambda d: d["name"])

    def __len__(self) -> int:
        return len(self._tools)
