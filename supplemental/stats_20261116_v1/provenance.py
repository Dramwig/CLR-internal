"""Read-only chronology, trainable parameter counts and evidence checks."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
from datetime import datetime
import torch
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from supplemental.stats_20261116_v1.runtime import digest

def main(c):
    torch.set_num_threads(2);root=Path(c['output']);used={}
    def read(p):
        p=Path(p);used[str(p)]=digest(p);return json.loads(p.read_text())
    m4=read('data/M4_v0/evaluation_suite.json');m41=read('data/M41_v1/training_complete.json');selection=read('data/M41_v1/variant_selection.json');params={}
    for model in ['v1','lite']:
        p=Path(m41['runs'][f'main_{model}_11'])/'best.pt';used[str(p)]=digest(p);state=torch.load(p,map_location='cpu',weights_only=True)
        # Frozen trainable_state stores trainable parameters plus normalization buffers.
        params[model]={'trainable_parameters':sum(v.numel() for k,v in state.items() if k not in ['state_mean','state_scale']),'checkpoint':str(p),'tensor_names':len(state)}
    ev=Path(m4['main_11'])/'events.jsonl';used[str(ev)]=digest(ev);events=[json.loads(x) for x in ev.read_text().splitlines()]
    old_input_manifests=['data/M10_v1/gsm_export_sources.json','data/M10b_v1/export_sources.json','data/M12_ablation/statistics_provenance.json']
    verified={}
    for path in old_input_manifests:
        d=read(path);sources=d.get('sha256',d.get('source_sha256',{}))
        for p,h in sources.items():
            if p in verified:
                assert verified[p]==h;continue
            if digest(p)!=h:raise ValueError('Frozen scientific input SHA mismatch '+p)
            verified[p]=h
    for p in ['docs/M4.1_TASK.md','reports/M4.1_details.md','scripts/m41_train_suite.py','docs/M5b_TASK.md','docs/M9_TASK.md']:
        used[p]=digest(p)
    write_json(root/'provenance.json',{'v0_evaluation_events':events,'v0_evaluation_run':m4['main_11'],'M41_selection':selection,'parameters':params,'selection_scope':'v1 versus lite; not the later single-step/default/joint trio','historical_test_exposure':True,'verified_scientific_input_files':len(verified),'checked_at':datetime.now().astimezone().isoformat(),'input_sha256':used})
    write_json(root/'frozen_scientific_input_sha256.json',verified)
    print('Parameter counts',params,'scientific files verified',len(verified),flush=True)

if __name__=='__main__':main(load_config(sys.argv[1]))
