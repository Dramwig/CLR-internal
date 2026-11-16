"""Paired training-group uncertainty for the preregistered projection check."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import numpy as np
import torch
from clr.audit.m71_runtime import snapshot
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded


def summary(values,draws):
    return {'mean':float(np.mean(values)),'lo':float(np.quantile(draws,.025)),'hi':float(np.quantile(draws,.975))}


def main(c):
    with recorded(c) as run:
        snapshot(run);root=Path(c['output']);selection=json.loads((root/c['selection_name']).read_text());source=Path(selection['run'])
        data=[torch.load(source/(kind+'_probe.pt'),weights_only=True,map_location='cpu')
              for kind in ['combined_unprojected','combined_projected']]
        if not torch.equal(data[0]['valid_indices'],data[1]['valid_indices']):raise ValueError('Unpaired projection probes')
        ids=data[0]['valid_indices'].numpy()//2;groups=sorted(set(ids));lookup={g:i for i,g in enumerate(groups)}
        if len(groups)!=c['expected_groups']:raise ValueError('Projection heldout group count changed')
        rng=np.random.default_rng(c['seed']);w=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=10000).astype(float)
        points=[];auc_draws=[];aucs=[]
        for item in data:
            logits=item['logits'][:,0];yy=item['truth'];nn=item['negative'];ix=torch.arange(len(yy))
            p=logits[ix,yy+4].numpy();n=logits[ix,nn+4].numpy()
            correct4=(logits[:,:4].argmax(1)==yy).numpy()
            correct1=((p>0).astype(float)+(n<=0).astype(float))/2
            a4=np.array([correct4[ids==g].mean() for g in groups]);a1=np.array([correct1[ids==g].mean() for g in groups])
            points.append({'A1_accuracy':a1,'A4_accuracy':a4})
            pos=np.stack([p[ids==g] for g in groups]);neg=np.stack([n[ids==g] for g in groups])
            diff=pos[:,:,None,None]-neg[None,None,:,:]
            matrix=((diff>0)+.5*(diff==0)).mean((1,3));aucs.append(float(matrix.mean()))
            auc_draws.append(np.einsum('bi,ij,bj->b',w,matrix,w,optimize=True)/len(groups)**2)
        result={'source_run':str(source),'groups':len(groups),'resampling_unit':'paired original/CF group','repeats':10000,
                'test_opened':False,'point_gate_below_2pp':selection['projection_loss_below_2pp'],'loss':{}}
        for metric in ['A1_accuracy','A4_accuracy']:
            delta=points[0][metric]-points[1][metric];draws=w@delta/len(groups)
            result['loss'][metric]=summary(delta,draws)
        result['loss']['A1_AUROC']=summary([aucs[0]-aucs[1]],auc_draws[0]-auc_draws[1])
        write_json(root/'projection_intervals.json',result)
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=False)
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))
