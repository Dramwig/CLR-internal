"""Post-smoke provisional capacity estimate; does not freeze or authorize data."""
import argparse,json
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
with recorded(c) as run:
 m=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
 if m.get('status')!='completed' or m['questions']!=200:raise ValueError('Successful 200-question smoke required')
 times=m['stage_seconds'];basic=(times['prepare']+times['early'])/200;traj=times['traj']/100;mc=times['mc']/100
 training=c['reader_train']+c['reader_val'];held=300+c['cal']+c['test']
 seconds=basic*(training+held)+traj*training+mc*6*held+mc*4*c['variance_pairs']
 disk=(2*m['storage']['reader_train']['bytes']/100*training+2*m['storage']['val']['bytes']/100*held+c['all_layers']*7*12*768*2*2)/2**30
 result={'status':'completed','provisional':True,'not_a_gate_approval':True,'requested_reader_train':c['reader_train'],'generation_gpu_hours_with_25pct_margin':seconds*1.25/3600,'estimated_output_gib_with_2x_storage_margin':disk,'scope':'full clean/all early exits;train K8;heldout six steps two K16 policies;50 pairs K64 both policies','excludes':'final measured gate cost; include all prior smoke/gate costs in definitive budget before freezing','gpu_hours':0,'source_forward_calls':0}
 run.write_metrics(**result);write_json(run.path/'estimate.json',result);print(run.path,result)
