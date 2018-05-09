import os
from ruamel.yaml import YAML
from tensorflow.contrib.training import HParams


class YParams(HParams):
    def __init__(self, dic):
        super().__init__()
        for yaml_fn, config_name in dic.items():
            with open(yaml_fn) as fp:
                for k, v in YAML().load(fp)[config_name].items():
                    self.add_hparam(k, v)


class Config(YParams):
    pass
