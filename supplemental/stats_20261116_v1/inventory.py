import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from clr.utils.config import load_config

def main(c):
    f=pd.read_csv('data/M12_ablation/costs.csv')
    f=f[(f['mode']=='full') & (f.stage=='train') & (f.status=='completed')]
    print('M12 costs',f.groupby('case').gpu_hours.sum().to_dict())
    count=json.loads(Path('data/M10_v1/exposure_targets.json').read_text())
    print('counts', {s:count[s]['11']['counts'] for s in count})
    for p in ['data/M10_v1/paper_inputs/main.csv','data/M9_v1/paper_inputs/main.csv','data/M10d_v1/B4_summary.csv']:
        g=pd.read_csv(p);print(p,list(g.columns),sorted(g.model.unique()))
    for p in ['data/M10_v1/paper_inputs/main_auroc_observations.csv','data/M12_ablation/auroc_observations.parquet']:
        if Path(p).exists():print('AUC',p)
    regs=json.loads(Path('data/M12_ablation/source_registry.json').read_text())
    print('first registry',regs[:2])
    print('B4 files',list(Path('data/M10d_v1').glob('*')))
    summary={'historical_M12_train_gpu_hours':f.groupby('case').gpu_hours.sum().to_dict(), 'counts':count}
    Path(c['output'],'inventory.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main(load_config(sys.argv[1]))
