"""Index exact input hashes, scripts and authorized paper version relationship."""
import argparse,hashlib,json
from pathlib import Path
import yaml
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(c):
 build=load_config(c['paper_build']);lines=['# Paper artifacts (M10 final)','','M10 updates the current paper; original M9 bytes and build remain recoverable via `data/M10_v1/m9_paper_snapshot/mapping.json`. Upstream freeze manifests are unchanged.','','Rebuild: `source scripts/m7_env.sh` then `.venv-paper/bin/python -I scripts/paper/make_all.py configs/M10/paper_build.yaml`. The clean plotting environment has no torch/transformers; dependency lock: `paper/requirements.lock.txt`.','','Every product has one entrypoint and YAML. Inputs are hash-frozen CSVs; scripts never call models. 95% CIs condition on fixed trained seeds, resampling entire problems/episode groups 10000 times.','','|Product|Entrypoint|YAML|Input CSV and SHA256|','|---|---|---|---|'];products={};records=[]
 for item in build['artifacts']:
  cfg=load_config(item['config']);inputs=[]
  for key,path in cfg['inputs'].items():inputs.append(f'`{path}` `{sha(path)}`')
  lines.append('|'+cfg['artifact']+'|`'+item['script']+'`|`'+item['config']+'`|'+'<br>'.join(inputs)+'|');records.append({'artifact':cfg['artifact'],'script':item['script'],'script_sha256':sha(item['script']),'config':item['config'],'config_sha256':sha(item['config']),'inputs':{p:sha(p) for p in cfg['inputs'].values()}})
 for sub in ['tables','figures','derived']:
  for p in sorted((Path('paper')/sub).glob('*')):
   if p.is_file():products[str(p)]=sha(p)
 lines+=['','T1 fills the LLaMA controls and adds B6; B5c Q3/Q4 is NR (12 GPUh budget admission failed), while Q1/Q2 are outside that baseline scope. T4 adds B6, with B5c audit A1/A4 NR. HiddenBench is a separate fact-panel/mismatch supplement, only 6 independent test groups; its 1% conformal threshold is infinite. The per-type appendix and F4 transfer curve retain negative/null results.','','T2 uses the same frozen deployed set definitions; unchanged figures use the exact M9 CSVs. F3 retains monotone envelopes and light raw points. TeX snippets use booktabs/longtable; no local LaTeX compiler is installed, so TeX compilation remains unperformed. PDFs are vector and use embedded fonts.','','Verification: `data/M10_v1/reproducibility.json`; integrity: `data/M10_v1/upstream_audit.json`; review: `reports/M10_REVIEW.md`; detailed statistics and provenance: `data/M10_v1/paper_inputs/`, `*export_sources.json`.','','## Version integrity','','The M9 freeze refers to original paper paths. M10 user authorization supersedes only current paper products and FINAL_SUMMARY; exact original bytes are preserved in the archive map. Integrity checks resolve authorized changed paths to their archived originals. All model/data/code upstream freezes are checked at their original paths.']
 Path('paper/README.md').write_text('\n'.join(lines)+'\n');manifests=[]
 for p in sorted(Path('data').rglob('freeze.json')):
  d=json.loads(p.read_text());entries=d.get('sha256',d.get('files',{}));manifests.append({'path':str(p),'sha256':sha(p),'entries':len(entries) if isinstance(entries,(list,dict)) else None})
 write_json('paper/artifact_manifest.json',{'version':'M10_paper_v1','current_freeze':'data/M10_v1/freeze.json','freeze_manifests':manifests,'artifacts':products,'builds':records,'source_provenance':'data/M10_v1/paper_inputs/provenance.json','integrity_audit':'data/M10_v1/upstream_audit.json','reproduction':'data/M10_v1/reproducibility.json','M9_archive_mapping':'data/M10_v1/m9_paper_snapshot/mapping.json','M9_archive_mapping_sha256':sha('data/M10_v1/m9_paper_snapshot/mapping.json'),'authorization':'M10 user explicitly requests current M9 paper tables/figures and FINAL_SUMMARY updates; old bytes preserved'})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))
