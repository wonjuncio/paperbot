"""Configuration management.

``Settings`` is a **metaclass-based singleton**: the first call to
``Settings.load()`` creates the instance; every later call returns
the same object.  Use ``update()`` to change paths at runtime, or
``reload()`` to re-read everything from disk.

All user-editable configuration lives under ``.metadata/``:

* ``email.yaml``         – Crossref polite-pool email
* ``llm_profiles.yaml``  – LLM provider credentials
* ``feeds.yaml``         – RSS journal feeds

On first run, missing files are copied from ``.metadata.example/``.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# LLM Profile dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMModel:
    """A single model entry from the built-in registry."""

    id: str
    name: str
    provider_id: str
    provider_name: str
    base_url: str
    context_window: int = 0
    max_output: int = 0


@dataclass
class LLMProfile:
    """A single LLM provider credential."""

    id: str
    name: str
    model: str
    api_key: str


@dataclass
class Feed:
    """A single RSS journal feed entry."""

    id: str
    name: str
    url: str
    issn: str = ""


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
    metadata_dir: Path = Path(".metadata")
    feeds_path: Path = Path(".metadata/feeds.yaml")
    export_dir: Path = Path("exports")

    # LLM profiles
    llm_profiles: list[LLMProfile] = field(default_factory=list)
    active_llm_id: Optional[str] = None

    # RSS feeds
    feeds: list[Feed] = field(default_factory=list)

    # ── Computed properties ────────────────────────────────────────────

    @property
    def active_llm(self) -> Optional[LLMProfile]:
        """Return the currently active LLM profile, or None."""
        if not self.active_llm_id:
            return None
        return next(
            (p for p in self.llm_profiles if p.id == self.active_llm_id),
            None,
        )

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

        metadata_dir = base_dir / ".metadata"
        cls._ensure_default_files(base_dir, metadata_dir)

        # Load email from .metadata/email.yaml
        contact_email = _load_email(metadata_dir / "email.yaml")

        # Load LLM profiles from .metadata/llm_profiles.yaml
        llm_profiles, active_llm_id = _load_llm_profiles(
            metadata_dir / "llm_profiles.yaml"
        )

        # Load RSS feeds from .metadata/feeds.yaml
        feeds = _load_feeds_as_objects(metadata_dir / "feeds.yaml")

        return cls(
            contact_email=contact_email,
            db_path=base_dir / "papers.db",
            metadata_dir=metadata_dir,
            feeds_path=metadata_dir / "feeds.yaml",
            export_dir=base_dir / "exports",
            llm_profiles=llm_profiles,
            active_llm_id=active_llm_id,
            feeds=feeds,
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
    def _ensure_default_files(base_dir: Path, metadata_dir: Path) -> None:
        """Copy ``.metadata.example/`` templates when real files are missing.

        Ensures ``.metadata/`` and its YAML files exist on first run by
        copying from ``.metadata.example/``.
        """
        metadata_dir.mkdir(parents=True, exist_ok=True)

        example_dir = base_dir / ".metadata.example"
        if not example_dir.exists():
            return

        for example_file in example_dir.iterdir():
            if example_file.is_file():
                target = metadata_dir / example_file.name
                if not target.exists():
                    shutil.copy2(example_file, target)
                    print(f"[PaperBot] Created .metadata/{example_file.name} from template")


# ---------------------------------------------------------------------------
# YAML loaders
# ---------------------------------------------------------------------------

def _load_email(path: Path) -> Optional[str]:
    """Load contact email from ``email.yaml``."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            email = data.get("contact_email")
            return email if email else None
    except Exception:
        pass
    return None


def save_email(path: Path, email: Optional[str]) -> None:
    """Persist contact email to ``email.yaml``."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Crossref API 접근용 이메일 (polite pool 사용을 위해 권장)\n")
        f.write("# 이메일 주소 입력 (예: you@example.com)\n")
        yaml.dump(
            {"contact_email": email or ""},
            f,
            default_flow_style=False,
            allow_unicode=True,
        )


def _load_llm_profiles(path: Path) -> tuple[list[LLMProfile], Optional[str]]:
    """Load LLM profiles from ``llm_profiles.yaml``.

    Returns:
        Tuple of (profiles list, active profile id)
    """
    if not path.exists():
        return [], None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return [], None

        active_id = data.get("active") or None
        raw_profiles = data.get("profiles") or []
        profiles = []
        for p in raw_profiles:
            if isinstance(p, dict) and p.get("id") and p.get("model") and p.get("api_key"):
                profiles.append(
                    LLMProfile(
                        id=str(p["id"]),
                        name=str(p.get("name", p["model"])),
                        model=str(p["model"]),
                        api_key=str(p["api_key"]),
                    )
                )
        return profiles, active_id
    except Exception:
        return [], None


def save_llm_profiles(
    path: Path,
    profiles: list[LLMProfile],
    active_id: Optional[str] = None,
) -> None:
    """Persist LLM profiles to ``llm_profiles.yaml``.

    Writes a clean YAML with comments matching the example template.
    """
    data: dict[str, Any] = {
        "active": active_id,
        "profiles": [
            {
                "id": p.id,
                "name": p.name,
                "model": p.model,
                "api_key": p.api_key,
            }
            for p in profiles
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write("# LLM 프로필 설정\n")
        f.write("# active: 현재 사용 중인 프로필 ID\n")
        f.write("# profiles: LLM 프로필 목록 (id, name, model, api_key)\n\n")
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def load_llm_models() -> list[LLMModel]:
    """Load the built-in LLM model registry from ``paperbot/data/llm_models.yaml``.

    This is **application data** (ships with the package), not user config.
    The UI ``<select>`` dropdown and API dispatch logic use this list.
    """
    registry_path = Path(__file__).resolve().parent / "data" / "llm_models.yaml"
    if not registry_path.exists():
        return []
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return []

        models: list[LLMModel] = []
        for provider in data.get("providers") or []:
            pid = provider.get("id", "")
            pname = provider.get("name", "")
            base_url = provider.get("base_url", "")
            for m in provider.get("models") or []:
                models.append(
                    LLMModel(
                        id=str(m["id"]),
                        name=str(m.get("name", m["id"])),
                        provider_id=pid,
                        provider_name=pname,
                        base_url=base_url,
                        context_window=int(m.get("context_window", 0)),
                        max_output=int(m.get("max_output", 0)),
                    )
                )
        return models
    except Exception:
        return []


def load_feeds(feeds_path: Path) -> list[dict[str, Any]]:
    """Load feed configurations from YAML file."""
    with open(feeds_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("feeds", [])


def _load_feeds_as_objects(path: Path) -> list[Feed]:
    """Load RSS feeds from ``feeds.yaml`` as :class:`Feed` objects.

    Each feed is assigned a stable short ID derived from a UUID.
    """
    import uuid

    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return []

        raw_feeds = data.get("feeds") or []
        feeds: list[Feed] = []
        for entry in raw_feeds:
            if isinstance(entry, dict) and entry.get("name") and entry.get("url"):
                feeds.append(
                    Feed(
                        id=str(uuid.uuid4())[:8],
                        name=str(entry["name"]),
                        url=str(entry["url"]),
                        issn=str(entry.get("issn", "")),
                    )
                )
        return feeds
    except Exception:
        return []


def save_feeds(path: Path, feeds: list[Feed]) -> None:
    """Persist RSS feeds to ``feeds.yaml``."""
    data: dict[str, Any] = {
        "feeds": [
            {
                "name": f.name,
                "url": f.url,
                "issn": f.issn,
            }
            for f in feeds
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
