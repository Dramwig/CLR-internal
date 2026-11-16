"""Rebuild T6 from frozen CSVs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from clr.paper.render_m10c import main
if __name__=='__main__':main()
