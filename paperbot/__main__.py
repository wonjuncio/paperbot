"""Entry point for running paperbot as a module or installed script.

Usage:
    paperbot / python -m paperbot         → GUI (streamlit run)
    paperbot <command> ... / python -m paperbot <command> ... → CLI
"""

import atexit
import sys
import uvicorn


def _reset_picked_on_exit() -> None:
    """Reset all is_picked to 0 when the process exits."""
    try:
        from paperbot.config import Settings
        from paperbot.database.repository import PaperRepository
        settings = Settings.load()
        repo = PaperRepository(settings.db_path)
        repo.reset_all_picked()
    except Exception:
        pass  # avoid breaking process exit


def run() -> None:
    """Entry point: no args → GUI (via streamlit run), else → CLI."""
    atexit.register(_reset_picked_on_exit)
    if len(sys.argv) == 1:
        uvicorn.run("paperbot.gui.app:app", host="127.0.0.1", port=8000, reload=True)
    else:
        from paperbot.cli import run_cli
        run_cli()


if __name__ == "__main__":
    run()
