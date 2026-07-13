import os
from pathlib import Path

from electrolyzer.core.bert import BERT


os.chdir(Path(__file__).parent)
config_fpath = Path(__file__).parent / "bert_config.yaml"
bert = BERT(config_fpath)
bert.run()
