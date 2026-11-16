from pathlib import Path
import ast
from scripts.m10b_resource_wait_v2 import eligible_devices

def test_startup_requires_stable_free_memory_and_resets():
 stable={};g=[(0,23000,50),(1,10000,100)]
 assert eligible_devices(g,stable,22528,3)==[]
 assert eligible_devices(g,stable,22528,3)==[]
 assert eligible_devices(g,stable,22528,3)==[g[0]]
 assert eligible_devices([(0,22000,99),(1,10000,100)],stable,22528,3)==[]
 assert stable[0]==0

def test_runtime_has_no_memory_stop_guard():
 tree=ast.parse(Path('scripts/m10b_controller_v3.py').read_text())
 calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='run']
 training=[n for n in calls if any(isinstance(x,ast.Constant) and x.value=='scripts/m10b_train_b5c.py' for x in ast.walk(n))]
 assert len(training)==1
 assert any(isinstance(n,ast.Constant) and n.value=='min_free_GiB=0' for n in ast.walk(training[0]))
 source=Path('scripts/m10b_resource_wait_v2.py').read_text()
 assert 'os.kill' not in source and 'signal.' not in source
