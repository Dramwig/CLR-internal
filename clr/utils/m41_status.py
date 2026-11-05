"""Persist milestone transitions to AGENTS and docs without modifying M4 status."""
import re
from pathlib import Path
from datetime import datetime,timezone,timedelta
from clr.utils.idea_a import write_json

def update(stage,details,**metadata):
    now=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    write_json(Path('docs/M41_LIVE.json'),{'updated_beijing':now,'stage':stage,'details':details,**metadata})
    line=f'M4.1（{now}）：{details}；状态索引docs/M41_LIVE.json。'
    p=Path('AGENTS.md');text=p.read_text();block='<!-- M41_LIVE -->\n'+line+'\n<!-- /M41_LIVE -->'
    text=re.sub(r'<!-- M41_LIVE -->.*?<!-- /M41_LIVE -->',lambda _:block,text,flags=re.S) if '<!-- M41_LIVE -->' in text else text+'\n'+block+'\n';p.write_text(text)
    Path('docs/STATUS.md').open('a').write('\n'+line+'\n')
