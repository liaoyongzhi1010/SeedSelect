import os
import yaml


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_data_paths(default_path=None):
    if default_path is None:
        default_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_paths.yaml')
    default_path = os.path.abspath(default_path)
    return load_yaml(default_path)
