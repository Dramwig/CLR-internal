"""Problem identity, not latent-step/sample index, controls membership."""
import hashlib,json
from pathlib import Path

def problem_id(question):
    return hashlib.sha256(' '.join(question.split()).encode()).hexdigest()

def freeze(path,splits):
    seen=set()
    for name,ids in splits.items():
        if len(set(ids))!=len(ids): raise ValueError(f'Duplicate problem IDs in {name}')
        if seen.intersection(ids): raise ValueError(f'Problem leakage in {name}')
        seen.update(ids)
    content=json.dumps(splits,sort_keys=True,indent=2)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_text()!=content: raise ValueError('Frozen split differs; create a new version')
    else:
        with path.open('x') as f:f.write(content)
    return hashlib.sha256(content.encode()).hexdigest()
