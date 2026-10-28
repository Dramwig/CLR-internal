"""CPU-only validation of actual official tokenizer and native harness prompts."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from scripts.m0b_huginn_eval import make_task,sha


def main(c):
    with recorded(c) as run:
        for source in (Path(__file__),Path('scripts/m0b_huginn_eval.py')):
            (run.path/source.name).write_text(source.read_text())
        task=make_task(c,run)
        tokenizer=AutoTokenizer.from_pretrained(c['snapshot'],trust_remote_code=True,local_files_only=True)
        def render(messages,add_generation_prompt=True):
            return tokenizer.apply_chat_template(messages,tokenize=False,
                add_generation_prompt=add_generation_prompt,continue_final_message=not add_generation_prompt)
        task.set_fewshot_seed(c['seed'])
        task.build_all_requests(system_instruction=c['system_instruction'],apply_chat_template=True,
            fewshot_as_multiturn=True,chat_template=render,tokenizer_name=str(c['snapshot']))
        records=[]
        for inst in task.instances:
            prompt=inst.args[0]
            ids=tokenizer(prompt,add_special_tokens=False)['input_ids']
            if c['system_instruction'] not in prompt or prompt.count('Q:')!=9:
                raise ValueError('Expected system instruction plus eight exemplars and target question')
            records.append({'problem_id':inst.doc['id'],'prompt':prompt,'input_ids':ids,
                'tokens':len(ids),'generation_kwargs':inst.args[1]})
        write_json(run.path/'rendered_prompts.json',records)
        write_json(run.path/'tokenizer_files.json',
            {p.name:sha(p) for p in Path(c['snapshot']).glob('*token*.json')})
        run.write_metrics(status='completed',stage='cpu_protocol_render_only',
            n=len(records),prompt_tokens=[r['tokens'] for r in records],
            system_preserved=True,eight_shot=True,gpu_hours=0,source_forward_calls=0)
        print(run.path,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*')
    args=p.parse_args();main(load_config(args.config,args.overrides))
