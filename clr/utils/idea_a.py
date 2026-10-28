"""Read-only imports from the existing Idea A checkout (never copy its code)."""
import sys
from pathlib import Path
ROOT = Path("/home/me/PaperField/LatDefer")
if not (ROOT / "latdef/utils/run.py").is_file():
    raise FileNotFoundError(f"Required Idea A checkout missing: {ROOT}")
sys.path.insert(0, str(ROOT))
from latdef.utils.run import Run, collect_environment, write_json, git_metadata
from latdef.utils.seed import seed_everything
from latdef.utils.logging import JsonlLogger
