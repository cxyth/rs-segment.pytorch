import os
import yaml


def load_config(config_name="config", config_dir=""):
    if os.path.splitext(config_name)[1] == ".yml":
        filepath = os.path.join(config_dir, config_name)
    else:
        filepath = os.path.join(config_dir, config_name + ".yml")

    if not os.path.isfile(filepath):
        raise FileNotFoundError('config file {} was not found.'.format(filepath))
    with open(filepath, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_dir):
    filepath = os.path.join(config_dir, 'config.yml')
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
