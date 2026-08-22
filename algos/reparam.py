"""Shim for the submission's reparam=True path: re-export the group mapping."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "submission") not in sys.path:
    sys.path.insert(0, str(_REPO / "submission"))

from parallel_adam_submission import build_group_mapping  # noqa: F401
