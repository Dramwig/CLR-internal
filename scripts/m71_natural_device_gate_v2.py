"""Val-only, identical-pair replay before moving fixed natural diagnostics."""
import sys
sys.dont_write_bytecode = True
import argparse, json, time
from pathlib import Path
import torch
from clr.audit.m71_runtime import Cost, snapshot, assert_final_lock
from clr.data.m71 import examples
from clr.sources.kv_relay import KVRelay
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything, write_json
from clr.utils.records import recorded

def main(c):
    torch.set_num_threads(2); seed_everything(c['seed']); torch.cuda.set_device(c['device'])
    with recorded(c) as run:
        snapshot(run); cost = Cost(run, c); relay = None
        try:
            assert_final_lock(c)
            rows = [json.loads(x) for x in Path(c['reference']).open()][:2]
            assert len(rows) == 2
            by_id = {e.id:e for e in examples(c, 'reader_val')}
            batch = [by_id[r['id']] for r in rows]
            assert all(r['group_id'] == e.meta['group_id'] for r,e in zip(rows,batch))
            relay = KVRelay(c); results = {}; started = time.monotonic()
            for mode, key in [('direct','direct'),('cot','cot512')]:
                relay.score(batch, sender=True, answer_mode=mode)
                results[key] = relay.last_sender
            errors = []; max_abs = 0.
            for i,r in enumerate(rows):
                for key in results:
                    actual, expected = results[key][i], r[key]
                    max_abs = max(max_abs, max(abs(a-b) for a,b in zip(actual['probs'],expected['probs'])))
                    for field in ['pred','generated_ids','generated_text','commitment','correct']:
                        if actual[field] != expected[field]: errors.append([r['id'],key,field])
            passed = not errors and max_abs <= c['max_probability_error']
            result = {'passed':passed,'max_probability_error':max_abs,'mismatches':errors,
                      'seconds':time.monotonic()-started,'device':c['device'],
                      'ids':[r['id'] for r in rows],'outputs':results,'test_opened':False}
            write_json(run.path/'device_gate.json',result)
            write_json(Path(c['output'])/'natural_device_gate_v2.json',{'run':str(run.path),**result})
            cost.update(relay,status='completed' if passed else 'failed',test_opened=False)
            if not passed: raise RuntimeError('Natural diagnostic device replay failed')
        except Exception:
            cost.update(relay,status='failed'); raise

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))
