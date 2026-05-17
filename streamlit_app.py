"""Streamlit Cloud entrypoint for SeatSense AI.

Streamlit Cloud sometimes defaults to ``streamlit_app.py``. The main app lives
in ``app.py``, so this wrapper runs that file after making the repository root
explicitly importable.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

runpy.run_path(str(PROJECT_ROOT / "app.py"), run_name="__main__")
