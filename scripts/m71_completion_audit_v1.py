"""Read-only completion audit of the original M7.1 experiment and artifacts."""
import sys
sys.dont_write_bytecode=True
import argparse,json,math,re,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from clr.audit.features import sha256,make_projection
from clr.audit.m71_runtime import snapshot,assert_final_lock
from clr.audit.m71_storage import load_bundle
from clr.data.m7 import threshold,commitment_route,format_commitment
from clr.data.m71 import fixed_hash_subset
from clr.sources.kv_relay import verify_idea_a
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded

def read(path):return json.loads(Path(path).read_text())
def jsonl(path):return [json.loads(x) for x in Path(path).open()]
def closed(path):
    path=Path(path);m=read(path/'metrics.json');assert m['status']=='completed',str(path)
    for name in ['config.json','git.json','environment.json','code_sha256.json','events.jsonl']:
        assert (path/name).is_file(),str(path/name)
    return m

def main(c):
    torch.set_num_threads(2);started=time.monotonic();root=Path(c['output']);rep=root/'pool'
    with recorded(c) as run:
        snapshot(run);result={'run':str(run.path),'model_calls':0,'gpu_hours':0,'requirements':{}}
        def done(key,evidence,limitation=None):
            result['requirements'][key]={'execution':'verified','evidence':evidence}
            if limitation:result['requirements'][key]['scientific_limitation']=limitation
        assert_final_lock(c);verify_idea_a(c['idea_a'])
        expected={};counts={}
        for name in ['data/M7_v1/freeze.json','data/M6_v1/freeze.json',str(root/'freeze.json')]:
            spec=read(name);files=spec.get('sha256',spec.get('files'));counts[name]=len(files)
            for path,digest in files.items():
                path=str(Path(path).resolve())
                assert path not in expected or expected[path]==digest,path
                expected[path]=digest
        def check(item):
            path,digest=item;return None if sha256(path)==digest else path
        with ThreadPoolExecutor(max_workers=2) as pool:
            mismatch=[x for x in pool.map(check,expected.items()) if x]
        assert not mismatch,mismatch
        result['freeze_counts']=counts;result['unique_frozen_files_checked']=len(expected)
        done('environment_and_immutability',{'frozen_files':counts,'final_lock_files':len(read(root/'final_lock.json')['sha256']),'generator':verify_idea_a(c['idea_a'])})
        print('frozen files verified',len(expected),flush=True)
        projection=read(root/'projection.json');matrix=torch.load(root/'projection.pt',map_location='cpu',weights_only=True)
        assert matrix.shape==(2048,512) and matrix.dtype==torch.float32
        assert sha256(root/'projection.pt')==projection['sha256']
        assert torch.equal(matrix,make_projection(2048,512,20261107))
        assert (matrix.T@matrix-torch.eye(512)).abs().max()<1e-6
        smoke=read(root/'source_smoke.json');assert smoke['smoke_passed'] and smoke['seconds']<=900 and smoke['groups']<=.05*1000 and not smoke['test_opened']
        gradient=read(Path(smoke['run'])/'feature_gradient_check.json')
        assert gradient['finite'] and gradient['actual_sender_KV'] and gradient['K_gradient_L1']>0 and gradient['V_gradient_L1']>0
        done('A1_features_and_gradient',{'source':'clr/audit/features.py','projection':projection,'actual_sender_gradient':gradient,'unit_tests':'reports/M71_evidence/tests_resume_v20.log'})
        scan=read(root/'layer_selection_v2.json');assert scan['groups']==1000 and scan['fit_groups']==800 and scan['heldout_train_groups']==200 and not scan['test_opened']
        curve=pd.read_csv(Path(scan['run'])/'layer_curves.csv')
        assert set(curve.layer)==set(range(36)) and len(curve)==72
        for ext in ['png','pdf']:assert (Path(scan['run'])/('layer_curves.'+ext)).stat().st_size>0
        ranking=curve[curve.representation=='unprojected'].copy()
        ranking['rank_score']=(ranking.A4_accuracy+ranking.A1_AUROC)/2
        ranked=ranking.sort_values(['rank_score','layer'],ascending=[False,True]).layer.tolist()
        assert ranked==scan['ranking']
        layers=sorted((ranked[0]+9*i)%36 for i in range(4));assert layers==scan['layers']==[7,16,25,34]
        assert all(x<.02 for x in scan['projection_loss'].values())
        done('A2_layer_scan_and_projection',{'scan':str(root/'layer_selection_v2.json'),'layers':layers,'projection_point_losses':scan['projection_loss'],'intervals':str(root/'projection_intervals.json')},'The <2pp criterion is a point estimate; the A4 loss CI extends to 5.75pp.')
        done('A3_and_B_smoke_and_estimates',{'smoke':smoke,'storage_comparison':read(root/'storage_comparison.json'),'actual_frozen_bytes':read(root/'freeze.json')['artifact_bytes']})
        groups={};ids={};entries={};bundle_count=sequence_count=0
        for split,n in c['counts'].items():
            folder=rep/'source'/split;rows=jsonl(folder/'manifest.jsonl');assert len(rows)==n
            groups[split]={r['group_id'] for r in rows};ids[split]={r['id'] for r in rows};assert len(groups[split])==len(ids[split])==n
            assert all(not groups[split]&other for name,other in groups.items() if name!=split)
            entries[split]={r['id']:r for r in rows}
            styles=Counter()
            for index,r in enumerate(rows):
                assert r['cf_id']==r['id']+'-cf' and r['layers']==layers and r['K']=='post-RoPE'
                assert r['commitment_source']=='programmatic' and r['commitment']==format_commitment(r['id'],r['route'][0])
                assert commitment_route(r['commitment'])==r['route'][0] and r['route'][0]!=r['route'][1]
                assert r['e'][1]==(r['e'][0]+1)%4 and r['cohort']==[r['id'],r['cf_id']]
                assert r['projection_sha256']==projection['sha256']
                styles[re.sub(r'Route [ABCD]','Route _',r['commitment'])]+=1
                bundle=load_bundle(r['file']);assert len(bundle['messages'])==2
                controls={'truncate','zero_layer','noise'} if split in ['cal_A','cal_B','test'] else set()
                assert set(bundle['controls'])==controls and set(r['control_raw_norms'])==controls
                for saved in [*bundle['messages'],*bundle['controls'].values()]:
                    values=saved['values'];assert values.ndim==2 and values.shape[1]==512 and values.dtype==torch.float16
                    assert torch.isfinite(values).all() and saved['mask'].dtype==torch.bool
                    assert all(len(saved[k])==len(values) for k in ['mask','layers','segments','kinds','token_ranges'])
                    assert saved['token_ranges'].shape==(len(values),2)
                    assert set(saved['layers'].tolist())==set(layers) and set(saved['kinds'].tolist())<={0,1}
                    for layer in layers:assert int(((saved['layers']==layer)&(saved['kinds']==1)).sum())==8
                    sequence_count+=1
                bundle_count+=1
                if (index+1)%1000==0:print('bundles checked',split,index+1,n,flush=True)
            assert len(styles)==3
            print('split complete',split,n,flush=True)
        done('B_full_data_pairs_splits_controls',{'groups':{k:len(v) for k,v in groups.items()},'bundles':bundle_count,'feature_sequences':sequence_count,'all_bundles_loaded':True,'overlapping_groups':0,'raw_controls_implementation':'clr/audit/m71_controls.py'})
        receiver_counts={};natural_counts={}
        for split in ['cal_A','cal_B','test']:
            rows=jsonl(rep/'source'/split/'receiver.jsonl');assert len(rows)==300
            selected=set(fixed_hash_subset(sorted(groups[split]),300));assert {r['group_id'] for r in rows}==selected
            assert all(r['feature_replay_max_abs']==0 for r in rows);receiver_counts[split]=300
        for split,n in [('reader_val',100),('test',300)]:
            rows=jsonl(rep/'source'/split/'natural_commitment.jsonl');assert len(rows)==n
            assert {r['group_id'] for r in rows}==set(fixed_hash_subset(sorted(groups[split]),n))
            for r in rows:
                for mode in ['direct','cot512']:
                    out=r[mode];assert out['correct']==(out['pred']==entries[split][r['id']]['route'][0])
                    assert len(out['probs'])==4 and abs(sum(out['probs'])-1)<1e-5
                    assert isinstance(out['generated_ids'],list) and 'generated_text' in out
            natural_counts[split]=n
        done('B_receiver_and_natural_authorized_panels',{'receiver':receiver_counts,'natural':natural_counts,'authorization':'docs/M7.1_TASK.md additional authorization'})
        gate=read(rep/'C1.json');assert gate['passed'] and not gate['test_opened']
        for kind in ['linear','B3']:
            g=gate['results'][kind];closed(g['run'])
            assert g['reader_train_groups']==8000 and g['reader_val_groups']==1000
            assert g['metrics']['A4']['accuracy']>=.45 and g['metrics']['A1']['AUROC']>=.75
        c2=read(rep/'C2.json');closed(c2['run']);assert c2['passed'] and not c2['test_opened'] and not c2['reader_val_used_for_overfit_stopping']
        assert sum(x['n'] for x in c2['selected']['metrics'].values())==500 and c2['selected']['accuracy']>=.98
        done('C1_C2_and_fallback_rule',{'C1':gate,'C2_run':c2['run'],'C2_accuracy':c2['selected']['accuracy'],'C2_rows':500,'fallback_needed':False})
        thresholds_checked=0;exposures={};model_count=0
        for seed in [11,22,33]:
            b3=read(rep/'training'/f'B3_{seed}.json');clr=read(rep/'training'/f'CLR_{seed}.json')
            assert clr['selected_exposure']>=b3['exposure'];exposures[seed]={'B3_actual':b3['exposure'],'CLR_selected':clr['selected_exposure']}
            for kind in ['CLR','B3','linear','B0','supervised']:
                training=read(rep/'training'/f'{kind}_{seed}.json');closed(training['run']);env=read(Path(training['run'])/'environment.json')
                assert env['packages']['transformers']=='4.57.1' and env['executable'].endswith('/.venv-comm/bin/python')
                cfg=read(Path(training['run'])/'config.json');assert cfg['seed']==seed and cfg.get('train_group_limit') is None
                cal=read(rep/'calibration'/f'{kind}_{seed}.json');closed(cal['run']);assert not cal['test_opened']
                assert sha256(training['checkpoint'])==cal['checkpoint_sha256']
                scores=jsonl(Path(cal['run'])/'cal_B_scores.jsonl');clean=[r for r in scores if r['category']=='clean'];assert len(clean)==1000
                for key,values in cal['thresholds'].items():
                    for alpha in [.01,.05]:
                        expected_q=threshold([r[key] for r in clean],alpha)
                        assert abs(expected_q-values[str(alpha)])<1e-12;thresholds_checked+=1
                evaluated=read(rep/'evaluated'/f'{kind}_{seed}.json');closed(evaluated['run']);assert evaluated['test_opened']
                for split,n in [('reader_val',1000),('test',2000)]:
                    values=jsonl(Path(evaluated['run'])/(split+'_scores.jsonl'))
                    assert sum(r['category']=='clean' for r in values)==n
                    assert sum(r['category']=='counterfactual' for r in values)==n
                    if kind!='supervised':
                        assert all((Path(evaluated['run'])/f'{split}_{task}_{templates}.pt.gz').is_file() for task in ['A1','A2','A3','A4'] for templates in ['seen','heldout'])
                        assert set(cal['thresholds'])=={'1','4','16'}
                model_count+=1
            natural=read(rep/'natural'/f'CLR_{seed}.json');closed(natural['run'])
            rows=jsonl(Path(natural['run'])/'natural_probabilities.jsonl');assert len(rows)==4800
            assert len({(r['split'],r['group_id'],r['mode'],r['template_id']) for r in rows})==4800
        for kind,n in [('norm',1000),('text',300)]:
            cal=read(rep/'baselines'/(kind+'.json'));closed(cal['run']);assert cal['clean_calibration_groups']==n
            scores=jsonl(Path(cal['run'])/'cal_B_scores.jsonl');clean=[r for r in scores if r['category']=='clean'];assert len(clean)==n
            for alpha in [.01,.05]:assert abs(threshold([r['score'] for r in clean],alpha)-cal['thresholds']['score'][str(alpha)])<1e-12
            closed(read(rep/'baselines'/(kind+'_evaluated.json'))['run'])
        done('D_three_seeds_exposure_calibration_predictions',{'models':model_count,'exposure':exposures,'threshold_values_recomputed':thresholds_checked,'natural_A3_rows_per_seed':4800,'QA_templates':'4 seen + 2 held out','k':[1,4,16],'alpha':[.01,.05]})
        done('D_supervised_binary_comparator',{'implementation':'scripts/m71_train.py supervised branch + scripts/m71_evaluate.py supervised_scores','label_equivalence_verified_for_all_groups':'A3 Yes=clean, No=CF; detected score=P(No)','seeds':[11,22,33]},'The directly supervised model remained at chance. Its measured gap is not a valid estimate of the cost of avoiding attack-specific training, nor a learned performance upper bound.')
        table=pd.read_csv(root/'analysis_v3/detection_group_bootstrap.csv')
        assert set(table.panel)=={'full','matched_text300'}
        for split in ['reader_val','test']:
            for kind in ['CLR','B3','linear','B0','norm','supervised']:
                for alpha in [.01,.05]:assert len(table[(table.panel=='full')&(table.split==split)&(table.kind==kind)&(table.alpha==alpha)&(table.category=='counterfactual')])>=1
        review=Path('reports/M7.1_REVIEW.md');text=review.read_text()
        assert len(text.encode())<=8000 and re.findall(r'### \(([a-f])\)',text)==list('abcdef')
        assert len(re.findall(r'^\|---',text,re.M))==3
        assert '36 passed' in Path('reports/M71_evidence/tests_resume_v20.log').read_text()
        for file in ['counterfactual_detection.png','counterfactual_detection.pdf','qa_group_bootstrap.csv','qa_CLR_minus_B3.csv','counterfactual_CLR_paired_differences.csv','natural_commitment_distribution.csv','costs.csv']:
            assert (root/'analysis_v3'/file).stat().st_size>0
        done('report_statistics_tests_and_four_answers',{'review':str(review),'bytes':len(text.encode()),'sections':6,'tables':3,'tests':36,'bootstrap_repeats':10000,'statistics':str(root/'analysis_v3')},'FPR columns are independent empirical test FPRs under clean cal_B nominal thresholds, not thresholds retuned on test.')
        frozen=read(root/'freeze.json');assert frozen['gpu_hours']<=30 and frozen['artifact_bytes']<=15*2**30
        done('budget_and_optional_M6_C2',{'GPUh':frozen['gpu_hours'],'frozen_bytes':frozen['artifact_bytes'],'optional_M6_C2':'not run; not a dependency'})
        result.update(status='completed_protocol_audit_passed',scientific_success_not_required=True,seconds=time.monotonic()-started,
                      findings={'supervised_upper_bound':'not established; failed learning is reported','A2_A3':'near chance; reported','projection':'point gate passed; uncertainty reported','historical_M7_raw_equivalence':'missing artifact; not a new M7.1 execution requirement'})
        write_json(Path(c['audit_result']),result)
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=True,requirements=len(result['requirements']),checked_bundles=bundle_count)
        print('completion audit passed',len(result['requirements']),'requirements;',bundle_count,'bundles;',sequence_count,'sequences',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))
