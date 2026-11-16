"""Versioned copies of frozen entrypoints; changes only source/runtime plumbing."""
import sys
sys.dont_write_bytecode=True
import json,difflib
from pathlib import Path
import yaml
import pandas as pd
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
    root=Path(c['output']);pkg='supplemental.stats_20261116_v1'
    for name in ['train','evaluate']:
        p=Path(f'scripts/m12_{name}.py');original=p.read_text();s=original
        s=s.replace('from clr.utils.m12_runtime import',f'from {pkg}.runtime import')
        s=s.replace('from clr.models.m12 import make_model',f'from {pkg}.model_data import make_model')
        s=s.replace('from clr.data.m4.reader import PrefixBank,rows_for,targets',f'from {pkg}.model_data import PrefixBank,rows_for\nfrom clr.data.m4.reader import targets')
        s=s.replace('from clr.data.m4.reader import PrefixBank,rows_for',f'from {pkg}.model_data import PrefixBank,rows_for')
        s=s.replace("norm=np.load(Path(c['m3_baseline_run'])/'normalization.npz')","norm=np.load(c['normalization'])")
        s=s.replace('for held in [False,True]:','for held in [False]:')
        s='import sys\nsys.dont_write_bytecode=True\n'+s
        (root/f'b_{name}.py').write_text(s)
        (root/f'b_{name}_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile=str(p),tofile=str(root/f'b_{name}.py'))))
    base=load_config('configs/M12/base.yaml')
    # Drop obsolete, unused milestone keys rather than carrying old authorities.
    keys=['templates','model_path','max_length','encoder_lr','head_lr','epochs','patience','weight_decay','variant','batch_size','eval_batch','min_epochs','qtypes']
    b={k:base[k] for k in keys};b.update(root='data/dumps/codi_llama_M6_v1',qa_root='data/M6_v1/qa',dim=2048,normalization='data/M6_v1/llama_normalization.npz',output=str(root),reader_min_free_gib=0,total_budget_hours=25,deadline=c['deadline'],smoke_receipt=str(root/'b_smoke_passed.json'),eval_lock=str(root/'b_final_lock.json'))
    for a in ['A1','A2','A4']:
        for seed,device in [(11,'cuda:4'),(22,'cuda:2'),(33,'cuda:3')]:
            d={**b,'case':a,'ablation':a,'seed':seed,'reader_device':device,'mode':'full'}
            (root/f'b_{a}_{seed}.yaml').write_text(yaml.safe_dump(d,sort_keys=False))
        d={**b,'case':'smoke_'+a,'ablation':a,'seed':11,'reader_device':'cuda:4','mode':'smoke','epochs':1,'min_epochs':1}
        (root/f'b_smoke_{a}.yaml').write_text(yaml.safe_dump(d,sort_keys=False))
    count=json.loads(Path('data/M10_v1/exposure_targets.json').read_text())
    ratio=sum(count['llama']['11']['counts'].values())/sum(count['gpt2']['11']['counts'].values())
    f=pd.read_csv('data/M12_ablation/costs.csv');f=f[(f['mode']=='full')&(f.stage=='train')&(f.status=='completed')]
    costs=f.groupby('case').gpu_hours.sum();estimate={a:float(costs[a]*ratio*1.7) for a in ['A1','A2','A4']}
    info={'estimated_gpu_hours':sum(estimate.values())+1.5,'training_by_variant':estimate,'train_QA_ratio':ratio,'native_dimension_shared_GPU_safety_factor':1.7,'smoke_evaluation_failure_reserve_gpu_hours':1.5,'per_type_reuse_new_gpu_hours':0,'per_type_origin':'data/M10_v1/trained/llama_CLR-per-type_*_full.json; exact 8 epochs per type, verified before final reporting','decision':'Run all 3 missing variants; reuse matched-exposure per-type and single step','recorded_before_training':True}
    write_json(root/'b_budget_preestimate.json',info)
    with (root/'REPORT.md').open('a') as stream:stream.write('\n### B 开跑前成本估计\n\n'+json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(info,ensure_ascii=False),flush=True)

if __name__=='__main__':main(load_config(sys.argv[1]))
