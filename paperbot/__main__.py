"""Entry point for running paperbot as a module or installed script.

Usage:
    paperbot / python -m paperbot         → GUI (streamlit run)
    paperbot <command> ... / python -m paperbot <command> ... → CLI
"""

import subprocess
import sys
from pathlib import Path


def run() -> None:
    """Entry point: no args → GUI (via streamlit run), else → CLI."""
    if len(sys.argv) == 1:
        gui_script = Path(__file__).resolve().parent / "gui.py"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(gui_script),
                "--server.headless",
                "true",
            ],
            check=True,
        )
    else:
        from paperbot.cli import run_cli
        run_cli()


if __name__ == "__main__":
    run()
