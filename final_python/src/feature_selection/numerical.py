from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression


def select_features(X_train, y_train, X_test, k='all', score_func=f_regression):
    """
    Select k best features based on score function.
    :param X_train: train dataset
    :param y_train: target variable vector
    :param X_test: test dataset
    :param k: 'all' for all features or number of features to select
    :param score_func: function defining a strategy
    :return: transformed train and test data and SelectBest object
    """
    # Configure to select all features
    fs = SelectKBest(score_func=score_func, k=k)

    # Learn relationship from training data
    fs.fit(X_train, y_train)

    # Transform train input data
    X_train_fs = fs.transform(X_train)

    # Transform test input data
    X_test_fs = fs.transform(X_test)

    return X_train_fs, X_test_fs, fs
