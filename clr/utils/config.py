import yaml
from clr.utils.idea_a import ROOT
from latdef.utils.config import UniqueKeyLoader

def load_config(path, overrides=()):
    with open(path) as f:
        config = yaml.load(f, Loader=UniqueKeyLoader)
    for override in overrides:
        key, value = override.split("=", 1)
        parts = key.split("."); dest = config
        for part in parts[:-1]: dest = dest[part]
        if parts[-1] not in dest: raise KeyError(key)
        dest[parts[-1]] = yaml.safe_load(value)
    return config
