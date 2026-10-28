from pathlib import Path
import traceback
from datetime import datetime, timezone
from contextlib import contextmanager
from clr.utils.idea_a import Run, write_json, git_metadata, ROOT as IDEA_A
ROOT = Path(__file__).resolve().parents[2]

@contextmanager
def recorded(config):
    with Run(config, root=ROOT, run_root=ROOT/"runs") as run:
        write_json(run.path/"dependencies.json", {"idea_a": git_metadata(IDEA_A)})
        try:
            yield run
        except Exception as exc:
            (run.path/"traceback.txt").write_text(traceback.format_exc())
            with (ROOT/"docs/BLOCKERS.md").open("a") as f:
                f.write(f"\n## {datetime.now(timezone.utc).isoformat()} — {run.path.name}\n"
                        f"{type(exc).__name__}: {exc}\n完整 traceback: `{run.path}/traceback.txt`。未回退模型或设置。\n")
            raise
