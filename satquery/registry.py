"""
The predefined registry, assembled.

Kept in one place so the trace, the UI and the tests all see the same set, and
so "which tools exist" is answerable without importing the controller.
"""
from __future__ import annotations

from .tools.base import Registry
from .tools.multi_image import ChangeTool, CrossModalTool
from .tools.specialists import CaptionTool, GroundingTool, VQATool


def build_registry() -> Registry:
    r = Registry()
    for tool in (VQATool(), CaptionTool(), GroundingTool(), ChangeTool(), CrossModalTool()):
        r.register(tool)
    return r


def build_controller():
    from .controller.agent import Controller
    return Controller(build_registry())
