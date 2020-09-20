import json
from typing import Dict


def read_json(json_path: str) -> Dict:
    """
    Deserializes JSON file.
    :param json_path: full path to JSON file
    :return: Python dictionary
    """
    with open(json_path, 'r') as read_file:
        return json.load(read_file)


def write_json(data: Dict, json_path: str) -> None:
    """
    Serializes a dictionary into JSON file.
    :param data: Python dictionary
    :param json_path: full path to JSON file
    :return: None
    """
    with open(json_path, "w") as write_file:
        json.dump(data, write_file)
