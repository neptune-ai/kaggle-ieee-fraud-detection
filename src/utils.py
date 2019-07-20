import yaml

from attrdict import AttrDict


def read_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f)
    return AttrDict(config)
