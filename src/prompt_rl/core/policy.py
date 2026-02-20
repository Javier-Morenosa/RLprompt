"""ActivePolicy — manages system_prompt.md as the Actor's live policy."""

from __future__ import annotations

from pathlib import Path


class ActivePolicy:
    """
    Thin wrapper around system_prompt.md.

    - read()  : hot-reloads the current prompt on every call (no caching).
    - write() : backs up the old version before overwriting.

    The file is the single source of truth for the Actor; the server reads it
    on every chat request so updates take effect immediately without a restart.
    """

    def __init__(
        self,
        path: str | Path,
        backup_dir: str | Path = "prompts",
        default: str = "",
    ) -> None:
        self.path       = Path(path)
        self.backup_dir = Path(backup_dir)
        self._default   = default

    def read(self) -> str:
        try:
            text = self.path.read_text(encoding="utf-8").strip()
            return text if text else self._default
        except FileNotFoundError:
            return self._default

    def write(self, prompt: str, version: int) -> None:
        """Backup the current file, then overwrite with the new prompt."""
        self.backup_dir.mkdir(exist_ok=True)
        ext = self.path.suffix or ".md"
        backup = self.backup_dir / f"prompt_v{version}{ext}"
        if self.path.exists():
            backup.write_text(
                self.path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        self.path.write_text(prompt.strip(), encoding="utf-8")
