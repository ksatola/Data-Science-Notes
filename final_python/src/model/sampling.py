from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler, ADASYN, SVMSMOTE
from imblearn.pipeline import make_pipeline


def get_oversampling_methods(random_state: int = 42) -> list:
    """
    Generates list of tuples with oversampling method name and instance for example
    ('SMOTE', SMOTE(**kwargs))
    :param random_state: random state for each of the oversampling methods
    :return: list of tuples
    """
    cls_oversampling_methods = [
        SMOTE(random_state=random_state),
        BorderlineSMOTE(random_state=random_state),
        RandomOverSampler(random_state=random_state),
        ADASYN(random_state=random_state),
        SVMSMOTE(random_state=random_state)
    ]

    oversampling_methods = []

    for method in cls_oversampling_methods:
        item = (type(method).__name__, method)
        oversampling_methods.append(item)

    return oversampling_methods
