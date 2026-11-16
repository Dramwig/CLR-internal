"""Val-only native dimension, masking, option identity and source parity gate."""
import sys
sys.dont_write_bytecode=True
import json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything,write_json
from clr.data.m12 import TensorSerializer
from clr.models.m6 import NativeLite
from supplemental.stats_20261116_v1.model_data import NativeAblation,PrefixBank,rows_for

def main(c):
    torch.set_num_threads(2);start=time.monotonic();ds=load_config(Path(c['output'])/'b_A1_11.yaml')
    rs=rows_for(ds,'reader_val','validate');ids=set(sorted({r['problem_id'] for r in rs})[:5]);rs=[r for r in rs if r['problem_id'] in ids];bank=PrefixBank(ds,'reader_val','validate',ids)
    s=TensorSerializer(AutoTokenizer.from_pretrained(ds['model_path'],local_files_only=True),load_config(ds['templates']))
    seed_everything(11);base=NativeLite(2048).eval();seed_everything(11);main=NativeAblation('main').eval()
    assert all(torch.equal(v,main.state_dict()[k]) for k,v in base.state_dict().items())
    out=[]
    with torch.no_grad():
        for q in ds['qtypes']:
            rr=[r for r in rs if r['qtype']==q][:8];z,m=bank.batch(rr,'cpu');text=s.batch(rr)
            a=base(text,z,m,q);b=main(text,z,m,q);assert torch.equal(a,b),(q,(a-b).abs().max())
            for ab in ['A1','A2','A4']:
                model=NativeAblation(ab).eval();p=model(text,z,m,q);zz=z.clone();zz[~m]=100000
                assert torch.equal(p,model(text,zz,m,q)),(q,ab,'padding')
                if q=='Q3':
                    perm=[2,0,3,1];rrp=[{**r,'options':[r['options'][i] for i in perm]} for r in rr]
                    assert torch.equal(p[:,perm],model(s.batch(rrp),z,m,q)),ab
            out.append({'qtype':q,'native_main_exact':True,'padding_and_option_alignment':True})
    receipt={'passed':True,'problems':len(ids),'elapsed_seconds':time.monotonic()-start,'test_opened':False,'checks':out}
    write_json(Path(c['output'])/'protocol_smoke.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main(load_config(sys.argv[1]))
