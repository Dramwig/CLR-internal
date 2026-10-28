import argparse
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import collect_environment, write_json
p=argparse.ArgumentParser();p.add_argument("config");p.add_argument("overrides",nargs="*");a=p.parse_args()
with recorded(load_config(a.config,a.overrides)) as run:
    e=collect_environment(".")
    import transformers, peft, tokenizers
    e['runtime_imports']={'transformers':transformers.__version__,'peft':peft.__version__,'tokenizers':tokenizers.__version__}
    write_json(run.path/"inventory.json",e)
    run.write_metrics(status="completed",kind="environment_inventory",gpu_hours=0,source_forward_calls=0)
    print(run.path)
