"""Review annotations and decimal rounding, then freeze the verified results."""
import sys
sys.dont_write_bytecode=True
import argparse,json,re,shutil
from decimal import Decimal,ROUND_HALF_EVEN
from pathlib import Path
import pandas as pd
import scripts.m71_natural_finish_v4 as annotations
import scripts.m71_freeze_v2 as freezer
from clr.audit.features import sha256
from clr.audit.m71_runtime import snapshot,assert_final_lock
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded

def sig(x):
    d=Decimal(str(round(float(x),12)))
    if not d:return '0'
    return format(d.quantize(Decimal(1).scaleb(d.adjusted()-2),rounding=ROUND_HALF_EVEN),'f')

def main(c):
    root=Path(c['output']);assert_final_lock(c)
    with recorded(c) as run:
        snapshot(run)
        path=Path('reports/M7.1_REVIEW.md')
        shutil.copyfile(path,root/'review_render_v3.md')
        annotations.original_execute=lambda *args,**kwargs:None
        annotations.execute('scripts/m71_freeze.py',c['freeze_config'],'manual_review_annotations')
        text=path.read_text();folder=root/'analysis_v3'
        det=pd.read_csv(folder/'detection_group_bootstrap.csv')
        qa=pd.read_csv(folder/'qa_group_bootstrap.csv');natural=pd.read_csv(folder/'natural_commitment_distribution.csv')
        kinds=['CLR','B3','linear','B0','supervised','norm','text']
        def row(split,kind,alpha=.05,panel='full'):
            k=16 if kind in ['CLR','B3','linear','B0'] else 0
            f=det[(det.split==split)&(det.kind==kind)&(det.alpha==alpha)&(det.panel==panel)&(det.category=='counterfactual')&(det.k==k)]
            return f.iloc[0] if len(f) else None
        def ci(r):return sig(100*r['mean'])+' ['+sig(100*r['lo'])+','+sig(100*r['hi'])+']'
        first='|方法|reader_val 检测% [CI] / FPR% / AUROC|test 检测% [CI] / FPR% / AUROC|\n|---|---|---|\n'
        for kind in kinds:
            cells=[]
            for split in ['reader_val','test']:
                r=row(split,kind)
                cells.append('未执行' if r is None else ci(r)+' / '+sig(100*r.FPR)+' / '+sig(r.AUROC))
            first+='|'+kind+'|'+'|'.join(cells)+'|\n'
        second='|QA|reader_val CLR/B3 %|test CLR/B3 %|test held-out CLR/B3 %|\n|---|---|---|---|\n'
        for task in ['A1','A2','A3','A4']:
            cells=[]
            for split,templates in [('reader_val','seen'),('test','seen'),('test','heldout')]:
                values=[]
                for kind in ['CLR','B3']:
                    r=qa[(qa.split==split)&(qa.kind==kind)&(qa.task==task)&(qa.templates==templates)&(qa.metric=='correct')].iloc[0]
                    values.append(sig(100*r['mean']))
                cells.append(' / '.join(values))
            second+='|'+task+'|'+'|'.join(cells)+'|\n'
        third='|自然承诺|reader_val n / P一致 [10%,90%]|test n / P一致 [10%,90%]|\n|---|---|---|\n'
        for mode in ['direct','cot512']:
            for correct in [True,False]:
                cells=[]
                for split in ['reader_val','test']:
                    f=natural[(natural.split==split)&(natural['mode']==mode)&(natural.model_correct==correct)]
                    if len(f):
                        r=f.iloc[0];cells.append(str(int(r.groups))+' / '+sig(r['mean'])+' ['+sig(r.q10)+','+sig(r.q90)+']')
                    else:cells.append('0 / NA')
                third+='|'+mode+(' 正确' if correct else ' 错误')+'|'+'|'.join(cells)+'|\n'
        for header,table in [('|方法|',first),('|QA|',second),('|自然承诺|',third)]:
            start=text.index(header);end=text.index('\n\n',start)
            text=text[:start]+table.rstrip()+text[end:]
        one='；'.join(kind+' '+sig(100*row('test',kind,.01)['mean'])+'% / FPR '+sig(100*row('test',kind,.01).FPR)+'%' for kind in ['CLR','B3','norm','text','supervised'])
        matched='；'.join(kind+' '+sig(100*row('test',kind,panel='matched_text300')['mean'])+'%' for kind in ['CLR','B3','norm','text','supervised'])
        text=re.sub(r'名义α=\.01的test CF检测：.*?。同一text300组的α=\.05检测：.*?；配对差值/CI及全部k、类别见CSV。',
                    '名义α=.01的test CF检测：'+one+'。同一text300组的α=.05检测：'+matched+'；配对差值/CI及全部k、类别见CSV。',text)
        receiver=json.loads((folder/'receiver_causal.json').read_text())
        values='/'.join(sig(100*receiver[k]['mean']) for k in ['clean','none','mismatch'])
        text=re.sub(r'有/无/错配准确率=[^%；]+%', '有/无/错配准确率='+values+'%',text)
        text=text.replace('自然承诺仅诊断，','natural的正确/错误sender分布高度重叠，未检出诚实推理错误。自然承诺仅诊断，')
        assert len(text.encode())<=8000
        assert re.findall(r'### \(([a-f])\)',text)==list('abcdef')
        assert len(re.findall(r'^\|---',text,re.M))==3
        path.write_text(text)
        details=Path('reports/M7.1_details.md');old=details.read_text()
        details.write_text('# M7.1 足量潜通信审计\n\n'+text+'\n\n统计与协议：'+old.split('统计与协议：',1)[1])
        metadata=root/'report.json';shutil.copyfile(metadata,root/'report_render_v3.json')
        report=json.loads(metadata.read_text());report.update(review_bytes=len(text.encode()),review_render='scripts/m71_review_freeze_v5.py',statistics='analysis_v3')
        write_json(metadata,report)
        write_json(root/'review_final_annotations_v5.json',{'review_bytes':len(text.encode()),'sha256':sha256(path),'decimal_rounding':'3 significant digits, half-even after 12-decimal numerical snap','prior_render':str(root/'review_render_v3.md')})
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=True,review_bytes=len(text.encode()))
    freezer.main(load_config(c['freeze_config']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))
