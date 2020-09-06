import json
from typing import Dict


def read_json(json_path: str) -> Dict:
    """
    Reads JSON file.
    :param json_path: full path to JSON file
    :return: Python dictionary
    """
    with open(json_path, 'r') as infile:
        return json.load(infile)
