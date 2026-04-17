"""
ClassGraph — the main graph data structure for Mosaic memory.

This module combines the three mixin modules into a single class:
- :class:`ClassGraphBase` — initialization, serialization, dual-graph sync, state tracking
- :class:`ClassGraphBuildMixin` — sensing, instance CRUD, edge building, conflict detection
- :class:`ClassGraphQueryMixin` — retrieval, neighbor expansion, keyword search

All external code should continue to import ``ClassGraph`` from this module.
"""
from src.data.graph_base import (  # noqa: F401 — re-export helpers
    ClassGraphBase,
    _instance_has_message_label,
    _message_label_key,
    _trim_build_context,
    _truncate_context,
)
from src.data.graph_build import ClassGraphBuildMixin
from src.data.graph_query import ClassGraphQueryMixin


class ClassGraph(ClassGraphQueryMixin, ClassGraphBuildMixin, ClassGraphBase):
    """Facade combining base, build, and query mixins."""
    pass
