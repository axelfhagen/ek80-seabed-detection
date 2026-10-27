"""Filesystem locations shared by scripts.

Defaults match the original cluster layout. Override them with the
``REPO_ROOT`` and ``DATA_ROOT`` environment variables when running elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("REPO_ROOT", CODE_ROOT.parent))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/cluster/work/axelfh"))
