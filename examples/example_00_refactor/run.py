import os
from pathlib import Path

from electrolyzer.core.bert import BERT
from electrolyzer.core.file_utils import load_yaml


os.chdir(Path(__file__).parent)
config_fpath = Path(__file__).parent / "bert_config.yaml"
bert = BERT(config_fpath)
bert.run()


# Run with recorder
config = load_yaml(config_fpath)
config["folder_output"] = Path(__file__).parent / "outputs"
config["recorder"] = {
    "flag": True,
    "file": "case.sql",
    "overwrite_recorder": True,
    "recorder_attachment": "model",
    "includes": ["*"],
}
bert = BERT(config)
bert.run()
