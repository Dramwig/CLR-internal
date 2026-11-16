"""Train-only 1k-group layer selection; original and projected probes are paired."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from clr.audit.features import KVFeatures,probe_summary
from clr.audit.m71_probes import fit_probes
from clr.audit.m71_runtime import Cost,snapshot,live
from clr.data.m71 import examples,scan_partition,selected_layers
from clr.data.m7 import counterfactual
from clr.sources.kv_relay import KVRelay
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.records import recorded
from scripts.m71_source import geometry,projection_file


def main(config):
    torch.set_num_threads(2);seed_everything(config['seed']);torch.cuda.set_device(config['device'])
    with recorded(config) as run:
        snapshot(run);cost=Cost(run,config);relay=None;live(stage='layer_scan',scan_run=str(run.path))
        try:
            root=Path(config['output']);smoke=json.loads((root/'source_smoke.json').read_text())
            if not smoke['smoke_passed']:raise ValueError('M7.1 source smoke not passed')
            free,total=torch.cuda.mem_get_info(config['device'])
            if free/2**30<config['min_free_GiB']:
                raise RuntimeError(f'Layer scan GPU free {free/2**30:.3g}GiB below configured requirement')
            matrix=projection_file(config);relay=KVRelay(config)
            layers=list(range(config['scan_layers']))
            transform=KVFeatures(matrix,layers,config['tail_tokens'],config.get('critical_tokens',False)).to(config['device'])
            items=list(examples(config,'reader_train',config['scan_groups']));groups=[e.meta['group_id'] for e in items]
            fitted,held=scan_partition(groups,config['scan_fit_groups'])
            # Memory-only unprojected summaries, not full raw KV or a disk dump.
            summaries=[];truth=[];negative=[];partition=[];metadata=[]
            for index,e in enumerate(items):
                cost.check();cf=counterfactual(e);batch=[e,cf];prefix=relay.encode(batch)
                for bi,item in enumerate(batch):
                    meta=geometry(item,relay.s2.tokenizer,prefix,bi)
                    feature=transform(prefix,prefix.mask,meta,bi,project=False)
                    summaries.append(probe_summary(feature).detach().cpu().half())
                    truth.append(item.meta['e_priv']);negative.append(batch[1-bi].meta['e_priv'])
                    partition.append(index in fitted)
                metadata.append({'id':e.id,'group_id':e.meta['group_id'],'fit':index in fitted,
                                 'sender_token_counts':[geometry(item,relay.s2.tokenizer,prefix,bi)['token_count'] for bi,item in enumerate(batch)]})
                del prefix
                if index%10==0:
                    cost.update(relay);write_json(run.path/'progress.json',{'stage':'sender_prefill','groups':index+1,'total':len(items),'gpu_hours':cost.elapsed()/3600})
                    print('scan_prefill',index+1,len(items),round(cost.elapsed(),2),flush=True)
            relay_calls=relay.forward_calls;relay_examples=relay.forward_examples;relay_tokens=relay.prefill_tokens
            del relay;relay=None;del transform;torch.cuda.empty_cache()
            x=torch.stack(summaries);del summaries
            write_json(run.path/'scan_groups.json',metadata)
            unprojected=fit_probes(x,truth,negative,partition,config,run,cost,'unprojected')
            xp=(x.float().reshape(len(x),len(layers),9,2048)@matrix).flatten(2).half()
            projected=fit_probes(xp,truth,negative,partition,config,run,cost,'projected')
            rank=sorted(layers,key=lambda l:(-(unprojected[l]['A4_accuracy']+unprojected[l]['A1_AUROC'])/2,l))
            chosen=selected_layers(rank[0],len(layers),4)
            if config.get('fixed_selection_file'):
                chosen=json.loads(Path(config['fixed_selection_file']).read_text())['layers']
            combined=[]
            for label,source in [('combined_unprojected',x),('combined_projected',xp)]:
                z=source[:,chosen].flatten(1)[:,None]
                combined.append(fit_probes(z,truth,negative,partition,config,run,cost,label)[0])
            gap={task:combined[0][task]-combined[1][task] for task in ['A1_accuracy','A4_accuracy']}
            frame=pd.DataFrame([{**row,'representation':name} for name,rows in [('unprojected',unprojected),('projected',projected)] for row in rows])
            frame.to_csv(run.path/'layer_curves.csv',index=False)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,2,figsize=(10,3.5))
            for ax,metric in zip(axes,['A4_accuracy','A1_AUROC']):
                for name in ['unprojected','projected']:
                    f=frame[frame.representation==name];ax.plot(f.layer,f[metric],label=name)
                for layer in chosen:ax.axvline(layer,alpha=.12,color='black')
                ax.set(xlabel='Qwen3 layer (zero based)',ylabel=metric);ax.legend()
            fig.tight_layout();fig.savefig(run.path/'layer_curves.png',dpi=160);fig.savefig(run.path/'layer_curves.pdf');plt.close(fig)
            result={'run':str(run.path),'groups':len(items),'fit_groups':len(fitted),'heldout_train_groups':len(held),
                    'ranking':rank,'layers':chosen,'selection_rule':'max mean(A4 accuracy,A1 AUROC), +9 modulo36; 4 layers',
                    'combined':{'unprojected':combined[0],'projected':combined[1]},
                    'projection_loss':gap,'projection_loss_below_2pp':all(v<.02 for v in gap.values()),
                    'test_opened':False}
            write_json(root/config.get('selection_name','layer_selection.json'),result)
            cost.update(status='completed',source_forward_calls=relay_calls,source_forward_examples=relay_examples,
                        source_prefill_tokens=relay_tokens,test_opened=False)
            live(stage='layer_scan_completed',scan_run=str(run.path),selected_layers=chosen,projection_loss=gap)
            print(json.dumps(result),flush=True)
        except Exception:
            cost.update(relay,status='failed');live(stage='layer_scan_failed',scan_run=str(run.path));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))
