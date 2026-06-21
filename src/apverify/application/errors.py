"""Errors a port may raise.

Defined in the application layer so use cases can react to a port failure without
importing any concrete adapter. Infrastructure adapters raise subclasses of
``PortError``; the dependency arrow stays pointed inward.
"""

from __future__ import annotations


class PortError(Exception):
    """Base class for failures surfaced through an application port."""
