"""Feature-only typed-question bank; no public/private source text enters readers."""
import json
import re
from pathlib import Path

import torch

from clr.audit.features import FeatureSequence,probe_summary
from clr.audit.m71_storage import load_bundle
from clr.audit.m7_reader import Serializer as M7Serializer
from clr.data.m7 import TEAMS,qa_rows,matched_false_values


def typed_condition(row):
    team=next((i for i,t in enumerate(TEAMS) if re.search(r'\b'+t+r'\b',row['question_text'])),0)
    route=re.search(r'Route ([ABCD])',row['question_text'])
    return team,ord(route[1])-65 if route else 0


class Serializer(M7Serializer):
    def batch(self,rows,device):
        # Assemble typed one-hot fields with the token batch on CPU, then
        # transfer once. Per-row CUDA indexing needlessly serializes kernels.
        batch=super().batch(rows,'cpu')
        team=[];route=[];option_teams=[]
        for index,row in enumerate(rows):
            t,r=typed_condition(row);team.append(t);route.append(r)
            option_teams.append([TEAMS.index(o) for o in row['options']] if row.get('task')=='A4' else [0]*len(row['options']))
            # These are deterministic encodings of literal words in the typed
            # question/options, available equally to every model. No labels.
            condition=torch.zeros(8)
            if row.get('task')=='A1':condition[t]=1
            if row.get('task') in ['A2','A3']:condition[4+r]=1
            question=(batch['blocks'][index]==0)
            batch['numeric'][index,question]=condition
            if row.get('task')=='A4':
                canonical=sorted(row['options'])
                for j,option in enumerate(canonical):
                    batch['numeric'][index,batch['markers'][index,j],TEAMS.index(option)]=1
        batch['team']=torch.tensor(team)
        batch['route']=torch.tensor(route)
        batch['option_teams']=torch.tensor(option_teams)
        return {key:value.to(device) for key,value in batch.items()}


class Bank:
    def __init__(self,root,split,templates=(0,1,2,3),representation='pool',load_sequences=True,limit=None):
        folder=Path(root)/representation/'source'/split
        if not (folder/'completed.json').exists():raise ValueError('Incomplete M7.1 split '+str(folder))
        self.entries=[json.loads(line) for line in (folder/'manifest.jsonl').open()]
        if limit is not None:self.entries=self.entries[:limit]
        self.rows=[];self.z={};self.pool={};self.source={};self.controls={}
        for entry in self.entries:
            bundle=load_bundle(entry['file'])
            for bi,mid in enumerate([entry['id'],entry['cf_id']]):
                saved=bundle['messages'][bi]
                sequence=FeatureSequence(**saved)
                self.pool[mid]=probe_summary(sequence).flatten().half()
                if load_sequences:self.z[mid]=saved
                self.source[mid]={'id':mid,'e':entry['e'][bi],'paired_e':entry['e'][1-bi],
                                  'route':entry['route'][bi],'paired_route':entry['route'][1-bi]}
            for kind,saved in bundle.get('controls',{}).items():
                mid=entry['id']+'::corrupted_'+kind
                self.pool[mid]=probe_summary(FeatureSequence(**saved)).flatten().half()
                if load_sequences:self.z[mid]=saved
                self.controls[mid]=kind
            base={'id':entry['id'],'group_id':entry['group_id']}
            for bi,category in enumerate(['clean','counterfactual']):
                mid=[entry['id'],entry['cf_id']][bi]
                self.rows+=qa_rows(base,self.source[mid],entry['commitment'],category,templates)
        self.layer_ids=self.entries[0]['layers']

    def pool_batch(self,rows,device):
        return torch.stack([self.pool[row['message_id']] for row in rows]).to(device)

    def add_mismatch_questions(self,templates=(0,1,2,3,4,5)):
        donors=self.entries[1:]+self.entries[:1]
        false=matched_false_values([self.source[donor['id']]['e'] for donor in donors])
        result=[]
        for entry,donor,negative in zip(self.entries,donors,false):
            if entry['group_id']==donor['group_id']:raise ValueError('Mismatch uses same group')
            source={**self.source[donor['id']],'paired_e':negative,'paired_route':donor['T'][donor['z_pub']][negative]}
            result+=qa_rows({'id':entry['id'],'group_id':entry['group_id']},source,entry['commitment'],'mismatch',templates)
        self.rows+=result
        return result

    def batch(self,rows,device):
        values=[self.z[row['message_id']] for row in rows];width=max(len(v['values']) for v in values)
        n=len(values);z=torch.zeros(n,width,512,dtype=torch.float16);mask=torch.zeros(n,width,dtype=torch.bool)
        layers=torch.zeros(n,width,dtype=torch.long);segments=layers.clone();kinds=layers.clone()
        for i,item in enumerate(values):
            size=len(item['values']);z[i,:size]=item['values'];mask[i,:size]=item['mask']
            layers[i,:size]=item['layers'];segments[i,:size]=item['segments'];kinds[i,:size]=item['kinds']
        return {key:value.to(device) for key,value in {'values':z,'mask':mask,'layers':layers,'segments':segments,'kinds':kinds}.items()}

    def pool_normalization(self):
        x=torch.stack(list(self.pool.values())).float()
        return x.mean(0),x.std(0).clamp_min(1e-4)

    def sequence_normalization(self):
        count=torch.zeros(36,dtype=torch.float64);sums=torch.zeros(36,512,dtype=torch.float64);squares=sums.clone()
        for entry in self.z.values():
            for layer in self.layer_ids:
                use=(entry['layers']==layer)&entry['mask'];v=entry['values'][use].double()
                count[layer]+=len(v);sums[layer]+=v.sum(0);squares[layer]+=v.square().sum(0)
        mean=sums/count.clamp_min(1)[:,None]
        scale=(squares/count.clamp_min(1)[:,None]-mean.square()).clamp_min(1e-8).sqrt()
        scale[count==0]=1
        return mean.float(),scale.float()
