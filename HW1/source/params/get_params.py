from yaml import safe_load
from typing import Dict, Any
from marshmallow_dataclass import class_schema

from source.params.pipe_params import PipeParams


def parse_yaml_file(yaml_path: str) -> Dict[Any, Any]:
    with open(yaml_path, "r") as f:
        config = safe_load(f.read())
    return config


def get_config(yaml_path: str, loaded_type=PipeParams) -> PipeParams:
    schema = class_schema(loaded_type)
    yaml_dict = parse_yaml_file(yaml_path)
    return schema().load(yaml_dict)
