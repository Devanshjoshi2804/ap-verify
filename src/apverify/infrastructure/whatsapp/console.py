"""A dry-run ``MessageSender`` that prints instead of sending.

Lets the collections agent be demonstrated end to end without messaging real
customers — the safe default for the CLI.
"""

from __future__ import annotations

from rich.console import Console


class ConsoleMessageSender:
    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def send(self, phone: str, text: str) -> str:
        self._console.print(f"  [cyan]→ {phone}[/cyan]  {text}")
        return "dry-run"
