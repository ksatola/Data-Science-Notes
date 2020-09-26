import pandas as pd
from utils import read_json, write_json
from pathlib import Path
from logger import logger

CONFIG_PATH = '../config/datasets.json'
CONFIG_PATH_BKP = '../config/datasets_bkp.json'


# https://machinelearningmastery.com/standard-machine-learning-datasets/
# https://realpython.com/python-json/


def get_dataset(name: str) -> pd.DataFrame:
    """
    Get a dataset by name.
    :param name: Predefined datasets name
    :return: Pandas DataFrame with the datasets data or null if not found
    """

    datasets = read_json(CONFIG_PATH)
    dataset = datasets[name]

    if dataset:
        if dataset['type'] == 'xls':
            return pd.read_excel(dataset['url'])
        elif dataset['type'] == 'csv':
            return pd.read_csv(dataset['url'], header=dataset['header'])
    else:
        return None


def add_dataset(data: pd.DataFrame,
                name: str,
                url: str = "",
                type: str = "csv",
                header: str = "infer",
                origin: str = "") -> str:
    """
    Add a dataset by name.
    :param data: Dataset in a form of Pandas DataFrame
    :param name: Unique dataset name
    :param url: Full path to save the dataset to (without file name)
    :param type: Type of the dataset (currently csv)
    :param header: "infer" or None
    :param origin: Url to source of the data
    :return: Dataset name added to JSON or None if error
    """

    # Read original JSON
    json_orig = read_json(CONFIG_PATH)

    # Check in JSON if name is unique
    if json_orig.get(name) is None:

        # Save original JSON in a different file as backup
        write_json(json_orig, CONFIG_PATH_BKP)

        # Save the dataset to url
        Path(url).mkdir(parents=True, exist_ok=True)
        path = Path(url).joinpath(f"{name}.{type}")
        if not path.exists():
            data.to_csv(path, index=False)
        else:
            logger.error(f"Dataset file already exists: {path}")

        # Save JSON with dataset metadata
        json_final = json_orig.copy()
        new_item = {"url": str(path), "type": type, "header": header, "origin": origin}
        json_final[name] = new_item
        write_json(json_final, CONFIG_PATH)

    else:
        logger.info("Dataset name already exists")
        return None


def add_prefix_to_list_items(prefix: str, items: list) -> list:
    """
    Add prefix to existing list items.
    :param prefix: a string of characters to be added at the beginning of each items item
    :param items: list with items to be renamed
    :return: list with renamed items
    """
    new_items = []
    for item in items:
        new_items.append(f"{prefix}{str(item)}")
    return new_items
