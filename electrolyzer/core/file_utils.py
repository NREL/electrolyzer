from pathlib import Path

import yaml


def load_yaml(filename, loader=yaml.SafeLoader) -> dict:
    if isinstance(filename, dict):
        return filename  # filename already yaml dict
    with Path.open(filename) as fid:
        return yaml.load(fid, loader)
