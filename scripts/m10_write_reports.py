"""Derive the M10 review and claim updates from final frozen CSV statistics."""
import argparse,json
from pathlib import Path
import pandas as pd
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m10_runtime import ledger

def pick(f,**kw):
 for k,v in kw.items():f=f[f[k]==v]
 if len(f)!=1:raise ValueError('Ambiguous/missing report statistic: '+str(kw))
 return f.iloc[0].to_dict()
def fmt(r,scale=1):return f'{r["mean"]*scale:.3g} [{r["lo"]*scale:.3g},{r["hi"]*scale:.3g}]'
def main(c):
 root=Path(c['output']);p=root/'paper_inputs';main=pd.read_csv(p/'main.csv');diff=pd.read_csv(p/'M10_gsm_paired.csv');qa=pd.read_csv(p/'communication_qa.csv');det=pd.read_csv(p/'communication_detection.csv');hb=pd.read_csv(p/'hiddenbench.csv');tr=pd.read_csv(p/'cross_use_transfer_paired.csv');costs=ledger();hours={s:sum(r['gpu_hours'] for r in costs if r['section']==s) for s in ['P1','P2-4','P2-5','P2-6']};total=sum(hours.values());repro=json.loads((root/'reproducibility.json').read_text());audit=json.loads((root/'upstream_audit.json').read_text())
 if not repro['passed'] or not audit['passed']:raise ValueError('Reproduction/integrity gate failed')
 def delta(source,split,baseline,q):return fmt(pick(diff,source=source,split=split,model='CLR-lite',baseline=baseline,qtype=q,metric='accuracy_difference'),100)
 def aqa(split,kind,t):return fmt(pick(qa,split=split,kind=kind,task=t,templates='seen',metric='correct'),100)
 def detection(split,kind,a,metric='detect'):
  r=pick(det,panel='full',split=split,kind=kind,k=16,alpha=a,category='counterfactual')
  return fmt(r,100) if metric=='detect' else f'{r["AUROC"]:.3g} [{r["AUROC_lo"]:.3g},{r["AUROC_hi"]:.3g}]'
 rows1=[]
 for source in ['gpt2','llama']:
  for split in ['reader_val','test']:
   rows1.append('|'+source+'/'+split+'|'+'|'.join(delta(source,split,b,'Q3')+' / '+delta(source,split,b,'Q4') for b in ['B3-prefix','B6','CLR-per-type'])+'|')
 rows2=[]
 for split in ['reader_val','test']:
  for kind in ['CLR','B6']:rows2.append('|'+split+'/'+kind+'|'+'|'.join([aqa(split,kind,'A1'),aqa(split,kind,'A4'),detection(split,kind,.05),detection(split,kind,.01),detection(split,kind,.05,'auc')])+'|')
 rows3=[]
 for kind in ['CLR','B3','B6','norm','text']:
  cells=[fmt(pick(hb,split='test',kind=kind,metric='detect_0.05'),100)]
  cells.extend(fmt(pick(hb,split='test',kind=kind,metric=t+'_accuracy'),100) if kind in ['CLR','B3','B6'] else 'NA' for t in ['A1','A4']);rows3.append('|'+kind+'|'+'|'.join(cells)+'|')
 transfer='；'.join('N='+n+' A1/A4 '+fmt(pick(tr,N=n,split='test',task='A1'),100)+' / '+fmt(pick(tr,N=n,split='test',task='A4'),100)+'pp' for n in ['100','300','1000','full'])
 b5=json.loads((root/'B5c_budget_admission.json').read_text());review=f'''# M10 审查

### (a) 状态一句话
M10预算范围内补充实验、论文产物及复现完成待审查，实验最终冻结；B5c全量准入失败，明确NR。

### (b) 关卡清单
- 通过：P1两源/三seed：LLaMA缺失控制、独立单层B6、曝光精确匹配的每类型CLR；按reader_val NLL选型，全部训练hash锁后最终test。
- 通过：val代理smoke、标签/划分/选项顺序/公开期望/conformal/组bootstrap测试；真实KV沿用M71四层与投影，Idea A只读。
- 通过：跨用途N100/300/1k/full、M9初始化对scratch、三seed；body与新adapter均微调，按标签数计N。
- 未通过（预算准入）：B5c最快实测仅GPT2 Q3/Q4一遍×三seed已需19.2 GPUh>12；未少seed/少标签替代，无全量比较。
- 通过/受限：HiddenBench65案例、63独立组全部使用，13/20/4/20/6组；仅mismatch。1%有限阈值未通过（cal_B20<99），+∞拒绝率0；6组test只作探索。
- 通过：累计{total:.3g}/40 GPUh；P1 {hours['P1']:.3g}/12，B5c {hours['P2-4']:.3g}/12，HiddenBench {hours['P2-5']:.3g}/12，迁移 {hours['P2-6']:.3g}/4。含失败/加载/校准/推断。
- 通过：{len(repro['artifacts'])}份产物独立无torch环境重建逐字节一致；上游{audit['unique_files']}文件SHA核验（授权paper更新解析到原字节归档）。

### (c) 核心主表（最多3张）
表1：CLR减对照，Q3 / Q4准确率差pp；固定三seed、10000题bootstrap 95%CI。完整Q1–Q4及绝对值见T1。

|源/集|CLR−prefix Q3 / Q4|CLR−B6 Q3 / Q4|多类型−单类型 Q3 / Q4|
|---|---|---|---|
{chr(10).join(rows1)}

表2：synthetic v2，准确率/检测率%，k16，95%组bootstrap CI；期望仅由公开commitment与约束推导。

|集/模型|A1|A4|CF@5%|CF@1%|AUROC|
|---|---|---|---|---|---|
{chr(10).join(rows2)}

表3：HiddenBench test6组，mismatch@5%及QA准确率%，95%组CI；所有模型@1%均为0（+∞阈值）。

|模型|Mismatch@5%|A1|A4|
|---|---|---|---|
{chr(10).join(rows3)}

迁移M9−scratch（test）：{transfer}。完整reader_val/test曲线、每类型比较见附表/F4。不能只据架构间差距证明“类型化接口”独立因果效应；注意力/容量/优化仍混杂。B5c无法比较。HiddenBench样本与任务适配均限制与synthetic结论的对应。

### (d) 本里程碑新增的偏离与阻塞
一次LLaMA补充吞吐smoke误用100/1000组=10%，保留并补50组；原20组关卡合规。B5c显存失败及本任务同卡调度失误导致prefix22失败，原记录计费；同科学配置重跑，后加逐GPU锁。
HiddenBench自由文本结论不可逆推出私有事实，统一程序化commitment显式声明被审计事实、原结论和公开one-of-panel约束；不构造CF。训练smoke按QA标签≤5%，最小train组1/13粒度限制明列；val1/20。
源smoke误用主环境dtype、汇总漏return及JSON分号解析失败均保留；独立通信环境/JSON字段解码修正后val通过。旧协议保留，公开契约v2对A1/A4输入、标签、期望与KV均无数值变更。
实时空闲GPU2/3接续未启动P1任务，保留原worker，科学设置与累计预算不变，交接有独立记录。M9 paper/FINAL_SUMMARY旧66文件逐字节归档，旧freeze不改。B5c及小HiddenBench限制不以缺失作零；未执行Huginn/Coconut或新攻击。

### (e) 需要审查者决定的问题
无新增授权问题；结论与限制待审查。

### (f) 证据路径
reports/M10_details.md；reports/FINAL_SUMMARY.md；paper/README.md、artifact_manifest.json；data/M10_v1/（final_lock.json、paper_inputs/*.csv、cost_ledger.json、reproducibility.json、upstream_audit.json、freeze.json、m9_paper_snapshot/mapping.json）；configs/M10/；reports/M10_evidence/；训练/预测run由各reference索引。
'''
 if len(review.encode())>8000:
  a=review.index('### (d)');b=review.index('### (e)');review=review[:a]+'''### (d) 本里程碑新增的偏离与阻塞
LLaMA一次吞吐smoke误用10%组，保留并补5%；原20组关卡合规。B5c OOM与同卡调度失误令prefix22失败，计费并同设置重跑，后加GPU锁。
HiddenBench仅63组；程序化事实+结论+one-of-panel公开契约，未构造CF，1%无穷阈值，test6组。训练smoke按标签≤5%，最小train组1/13；源环境/汇总/JSON标点错误保留并val修正。
P1后续任务分GPU2/3并行，原worker保留，科学设置不变。原M9 paper66文件归档，旧freeze不改；细节及失败run见M10_details/BLOCKERS。

'''+review[b:]
 if len(review.encode())>8000:raise ValueError('M10 review exceeds 8000 bytes: '+str(len(review.encode())))
 Path('reports/M10_REVIEW.md').write_text(review)
 details='''# M10详细证据

所有主表由data/M10_v1/paper_inputs/*.csv派生；每个原始预测文件SHA在*export_sources.json中。
统计为固定三个训练seed的条件CI，按problem/episode组bootstrap10000次，不能解释为重新训练seed的不确定性。
HiddenBench保留同公开描述变体于同一组；mismatch donor在同split异组，区间条件于固定配对；只有6个独立test组。
B6只用选项/查询事实的冻结文本embedding与数值特征；单层cross-attention和线性头，各类型独立，无共享模板接口。
每类型CLR全架构不变，精确匹配原多类型checkpoint该类型的8整epoch曝光；目标数据在exposure_targets.json。
迁移只复用M9 body，adapter随机且scratch配对相同初始化；两者都微调，同标签面板、LR和曝光，val100组选点后最终完整reader_val/test。
B5c代码/失败及成功smoke保留；完整评估没有执行，所有论文行NR，不能比较训练式语言化方法。
HiddenBench是事实候选面板适配，程序化commitment显式声明审计事实与exactly-one约束，不等价于一般结论语义审计。
Norm在未标注train拟合，text按M71离散可见结论/receiver选择特征在cal_A拟合C=1逻辑回归；cal_B独立组定阈值。
源GPU型号为RTX4090，实际型号/环境与每次代码快照位于run。物理source forward与token数保留在源run metrics。

## 用户五问的直接证据

'''
 details+='1. LLaMA控制前缀后Q3/Q4优势：'+delta('llama','test','B3-prefix','Q3')+' / '+delta('llama','test','B3-prefix','Q4')+'pp。\n'
 for source in ['gpt2','llama']:details+='2. '+source+' CLR−B6 Q3/Q4：'+delta(source,'test','B6','Q3')+' / '+delta(source,'test','B6','Q4')+'pp。\n'
 details+='3. B5c：未执行完整比较（预算）。\n4. 每类型比较见M10_gsm_paired.csv和appendix_per_type.tex，不能概括所有类型全胜。\n5. HiddenBench：mismatch和A1/A4，test仅6组；跨用途初始化差：'+transfer+'。\n\n## 审查正文\n\n'+review
 Path('reports/M10_details.md').write_text(details)
 # Update the exact archived M9 summary, retaining all six prior evidence lines.
 old=Path('data/M10_v1/m9_paper_snapshot/reports/FINAL_SUMMARY.md').read_text();old=old.replace('实验冻结至M9；除M9-A外未新增模型实验。','实验最终冻结至M10；M10只补授权的投稿对照，M9及上游原字节保留。').replace('## C1 类型化读取在状态与选项交互任务上优于所比较探针','## C1 状态与选项交互读取的架构比较（新增B6后按类型限定）')
 add='\n\n## M10新增证据与主张修订\n\n'
 add+='T1：LLaMA CLR−prefix Q3/Q4 '+delta('llama','test','B3-prefix','Q3')+' / '+delta('llama','test','B3-prefix','Q4')+'pp。\n\n'
 for source in ['gpt2','llama']:add+=source+' CLR−B6 Q3/Q4 '+delta(source,'test','B6','Q3')+' / '+delta(source,'test','B6','Q4')+'pp；多类型−单类型 '+delta(source,'test','CLR-per-type','Q3')+' / '+delta(source,'test','CLR-per-type','Q4')+'pp。\n\n'
 add+='这些差距属于整体架构比较，不能单独归因于类型化接口；B6单层注意力也能读取部分内容。T4：B6 A1/A4 '+aqa('test','B6','A1')+' / '+aqa('test','B6','A4')+'%；CF@5% '+detection('test','B6',.05)+'%。\n\n'
 add+='F4跨用途初始化：'+transfer+'。这是微调初始化比较，不是冻结主体只训适配器；是否有益须按N和类型分别说。\n\n'
 add+='HiddenBench：CLR/B6 mismatch@5% '+fmt(pick(hb,split='test',kind='CLR',metric='detect_0.05'),100)+' / '+fmt(pick(hb,split='test',kind='B6',metric='detect_0.05'),100)+'%。63独立组，test6组；程序化事实+one-of-panel约束，不能作为一般counterfactual审计的独立验证。1%阈值无穷，零拒绝不表明检测有效。\n\n'
 add+='B5c未通过12GPUh全量准入；不能声称CLR优于训练式语言化基线。每类型/多类型没有被预设为全胜；完整结果见附表。\n'
 final=old+add
 if len(final.encode())>10000:raise ValueError('FINAL_SUMMARY exceeds 10KB')
 Path('reports/FINAL_SUMMARY.md').write_text(final)
 claims=json.loads(Path('data/M10_v1/m9_paper_snapshot/paper/claims.json').read_text());claims['claims'][0]['claim']='状态与选项交互读取在既有B3比较有优势；B6与每类型对照按源/类型限定'
 claims['claims'][0]['limits']+=' M10 B6对照见新增配对CSV，不能把整体架构差距单独归因于类型化接口。'
 claims['M10_update']={'review':'reports/M10_REVIEW.md','gsm_paired':'data/M10_v1/paper_inputs/M10_gsm_paired.csv','audit_paired':'data/M10_v1/paper_inputs/M10_audit_paired.csv','cross_use':'data/M10_v1/paper_inputs/cross_use_transfer_paired.csv','limitations':['B6 comparison does not isolate causal contribution of typed interface','B5c full comparison unavailable under budget','HiddenBench only 6 independent test groups and fact-panel adaptation','Cross-use experiment fine-tunes initialized body; not adapter-only frozen transfer']};write_json('paper/claims.json',claims)
 print('review bytes',len(review.encode()),'summary bytes',len(final.encode()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))
