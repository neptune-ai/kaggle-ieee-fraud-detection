import os
import yaml

from attrdict import AttrDict


def read_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f)
    return AttrDict(config)


def check_env_vars():
    assert os.getenv('NEPTUNE_API_TOKEN'), """You must put your Neptune API token in the \
NEPTUNE_API_TOKEN env variable. You should run:
    $ export NEPTUNE_API_TOKEN=your_neptune_api_token"""
    assert os.getenv('CONFIG_PATH'), """You must specify path to the config file in \
CONFIG_PATH env variable. For example run:
    $ export CONFIG_PATH=config_kaggle.yml"""
