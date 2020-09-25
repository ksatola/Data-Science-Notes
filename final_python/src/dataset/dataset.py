import pandas as pd
from utils import read_json


def get_dataset(name: str) -> pd.DataFrame:
    """
    Get a dataset by name.
    :param name: Predefined dataset name.
    :return: Pandas DataFrame with the dataset data or null if not found.
    """

    datasets = read_json('../config/datasets.json')
    dataset = datasets[name]

    if dataset:
        if dataset['type'] == 'xls':
            return pd.read_excel(dataset['url'])
    else:
        return None
