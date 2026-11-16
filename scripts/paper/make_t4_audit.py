"""Render t4_audit from frozen CSV evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from clr.paper.render import main
if __name__ == "__main__":main('t4')
