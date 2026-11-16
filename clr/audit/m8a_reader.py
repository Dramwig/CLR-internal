"""Public-only differentiable audit scores for frozen M7.1 readers."""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from clr.audit.m71_reader import Serializer
from clr.audit.m71_evaluation import consistency_rows
from clr.audit.m8_p0 import public_constraints
from clr.models.m71 import Reader


class Auditors:
    def __init__(self,c):
        self.device=c['device'];self.models={};self.cal={}
        self.serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True))
        for seed in c['reader_seeds']:
            t=json.loads(Path(f'data/M71_v1/pool/training/CLR_{seed}.json').read_text())
            model=Reader('CLR').to(self.device)
            model.load_state_dict(torch.load(t['checkpoint'],weights_only=True,map_location='cpu'))
            self.models[seed]=model.eval().requires_grad_(False)
            self.cal[seed]=json.loads(Path(f'data/M71_v1/pool/calibration/CLR_{seed}.json').read_text())

    def questions(self,e,commitment,seed):
        public=public_constraints(e.x_pub)
        entry={'id':e.id,'group_id':e.meta['group_id'],'commitment':commitment,**public}
        # The called function reads no oracle e/route. Only public table, public
        # z, literal commitment, and question RNG enter expected answers.
        rows=consistency_rows({'entry':entry,'message_id':e.id},seed,16)
        return [(task,[r for r in rows if r['task']==task]) for task in ['A1','A2','A3','A4']]

    def prepare(self,e,commitment,seed):
        return [(task,self.serializer.batch(rows,self.device),torch.tensor([r['label'] for r in rows],device=self.device))
                for task,rows in self.questions(e,commitment,seed) if rows]

    def score(self,sequence,seed,prepared):
        losses=[]
        source={key:getattr(sequence,key)[None] for key in ['values','mask','layers','segments','kinds']}
        source['values']=source['values'].half()
        for task,text,target in prepared:
            n=len(target);batch={key:value.expand(n,*value.shape[1:]) for key,value in source.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=self.models[seed](text,batch,task)
            logp=torch.log_softmax(logits.float()/self.cal[seed]['temperatures'][task],-1)
            losses.append((-logp.gather(1,target[:,None]).squeeze(1)).clamp_max(27.631021115928547))
        return torch.cat(losses).mean()

    @torch.no_grad()
    def evaluate(self,sequence,prepared):
        out={}
        for seed in self.models:
            score=float(self.score(sequence,seed,prepared[seed]))
            out[str(seed)]={'score':score,**{str(alpha):score>self.cal[seed]['thresholds']['16'][str(alpha)] for alpha in [.01,.05]}}
        return out
