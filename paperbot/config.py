"""Configuration management.

``Settings`` is a **metaclass-based singleton**: the first call to
``Settings.load()`` creates the instance; every later call returns
the same object.  Use ``update()`` to change paths at runtime, or
``reload()`` to re-read everything from disk.
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Singleton metaclass
# ---------------------------------------------------------------------------

class _SettingsMeta(type):
    """Metaclass that enforces a process-wide singleton for *Settings*.

    * First ``Settings(...)`` creates and caches the instance.
    * Later ``Settings(...)`` calls return the cached instance (args ignored).
    """

    _instances: dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# ---------------------------------------------------------------------------
# Settings dataclass (singleton)
# ---------------------------------------------------------------------------

@dataclass
class Settings(metaclass=_SettingsMeta):
    """Application settings — singleton with runtime-mutable paths.

    Usage::

        settings = Settings.load()          # first call → create
        settings = Settings.load()          # later → same object
        settings.update(db_path=Path(...))  # runtime change
        settings = Settings.reload()        # re-read from disk
    """

    contact_email: Optional[str] = None
    db_path: Path = Path("papers.db")
    feeds_path: Path = Path("feeds.yaml")
    export_dir: Path = Path("exports")

    # ── Runtime helpers ───────────────────────────────────────────────

    def update(self, **kwargs: Any) -> None:
        """Mutate settings fields at runtime.

        >>> Settings.load().update(db_path=Path("/tmp/test.db"))
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"Settings has no field '{key}'")
            setattr(self, key, value)

    # ── Factory / lifecycle ───────────────────────────────────────────

    @classmethod
    def load(cls, base_dir: Optional[Path] = None) -> "Settings":
        """Load or return the singleton Settings instance.

        On first call the singleton is created; subsequent calls return
        the cached instance.  Pass *base_dir* to override the project
        root (defaults to the repository root one level above ``paperbot/``).
        """
        if cls in _SettingsMeta._instances:
            return _SettingsMeta._instances[cls]  # type: ignore[return-value]

        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent

        cls._ensure_default_files(base_dir)
        load_dotenv(base_dir / ".env")

        return cls(
            contact_email=os.getenv("CONTACT_EMAIL"),
            db_path=base_dir / "papers.db",
            feeds_path=base_dir / "feeds.yaml",
            export_dir=base_dir / "exports",
        )

    @classmethod
    def reload(cls, base_dir: Optional[Path] = None) -> "Settings":
        """Discard the current singleton and re-load from disk."""
        cls.reset()
        return cls.load(base_dir)

    @classmethod
    def reset(cls) -> None:
        """Discard the singleton so the next ``load()`` re-creates it."""
        _SettingsMeta._instances.pop(cls, None)

    # ── Private ───────────────────────────────────────────────────────

    @staticmethod
    def _ensure_default_files(base_dir: Path) -> None:
        """Copy *.example* templates when the real files are missing.

        Ensures ``feeds.yaml`` and ``.env`` exist on first run by
        copying from ``feeds.yaml.example`` / ``.env.example``.
        """
        for example_name, target_name in (
            ("feeds.yaml.example", "feeds.yaml"),
            (".env.example", ".env"),
        ):
            target = base_dir / target_name
            example = base_dir / example_name
            if not target.exists() and example.exists():
                shutil.copy2(example, target)
                print(f"[PaperBot] Created {target_name} from {example_name}")


# ---------------------------------------------------------------------------
# Feed loader
# ---------------------------------------------------------------------------

def load_feeds(feeds_path: Path) -> list[dict[str, Any]]:
    """Load feed configurations from YAML file."""
    with open(feeds_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("feeds", [])
