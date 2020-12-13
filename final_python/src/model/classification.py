from typing import Dict
from collections import Counter

import numpy as np
import os
import pandas as pd

from pickle import dump, load

#from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler, ADASYN, SVMSMOTE
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier


from sklearn.model_selection import train_test_split

from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import (
    SMOTE,
    BorderlineSMOTE,
    RandomOverSampler,
    ADASYN,
    SVMSMOTE,
    SMOTENC
)

from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
from sklearn.decomposition import PCA, TruncatedSVD, FastICA

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

from logger import logger


# from logger import choose_logger_type
# from .secondary_helpers import (
# prepare_data_for_classification,
# check_binary_model_performance_on_test
# )


def get_models_for_classification(random_state: int = 42) -> list:
    """
    Generates list of tuples with model name and instance for example
    ('LogisticRegression', LogisticRegression(**kwargs))
    :param random_state: random state for each of the oversampling methods
    :return: list of tuples
    """
    cls_models = [
        LogisticRegression(random_state=random_state),
        SGDClassifier(random_state=random_state),
        DecisionTreeClassifier(random_state=random_state),
        KNeighborsClassifier(),
        AdaBoostClassifier(random_state=random_state),
        BaggingClassifier(random_state=random_state, n_estimators=10),
        RandomForestClassifier(random_state=random_state, n_estimators=10),
        ExtraTreesClassifier(random_state=random_state, n_estimators=10),
        SVC(random_state=random_state, gamma='scale'),
        CatBoostClassifier(iterations=1000, learning_rate=0.01, l2_leaf_reg=3.5, depth=2,
                           rsm=0.98, use_best_model=False, random_seed=random_state,
                           logging_level='Silent')
    ]

    models = []

    for model in cls_models:
        item = (type(model).__name__, model)
        models.append(item)

    return models

def check_classification_performance(data: pd.DataFrame,
                                     target_name: str,
                                     modeling_config: Dict,
                                     model_type: str,
                                     models: list,
                                     oversampling_methods: list,
                                     oversample: bool = True,
                                     output_directory: str = "/results/model/") -> pd.DataFrame:
    """
    Creates and saves a data frame with modeling outcomes for a chosen target variable
    :param data: Pandas DataFrame
    :param target_name: name of a target variable
    :param modeling_config: JSON config file for modeling
    :param model_type: type of the model: "binary_classification" or "multiclass_classification"
    :param models: list of tuples with models names and instances for example
                   [('LogisticRegression', LogisticRegression(**kwargs)),
                   ('SGDClassifier', SGDClassifier(**kwargs))],
                   list of proposed models is saved in get_models_for_classification() function
    :param oversampling_methods: list of tuples with oversampling methods names and instances for example
                   [('SMOTE', SMOTE(**kwargs)),
                   ('ADASYN', ADASYN(**kwargs))],
                   list of proposed oversampling methods is saved in
                   get_oversampling_methods() function
    :param oversample: boolean variable if oversampling should be performed or not
    :param output_directory: directory for saving the data frame with outcomes from the modeling
    :return: pandas data frame with outcomes from the modeling
    """

    scaler_methods = modeling_config["scaler_methods"]
    #scaler_methods.append(None)
    dimensionality_reduction_methods = modeling_config["dimensionality_reduction_methods"]
    #dimensionality_reduction_methods.append(None)

    if not oversample:
        oversampling_methods = [None]
    else:
        oversampling_methods = oversampling_methods  # modeling_config["oversampling_methods"]
        #oversampling_methods.append(None)

    full_output = []

    for scaler in scaler_methods:
        for dim_red_method in dimensionality_reduction_methods:
            X_train, X_test, y_train, y_test = prepare_data_for_classification(
                data=data,
                target_name=target_name,
                modeling_config=modeling_config,
                model_type=model_type,
                oversampling_method=None,
                undersampling_method="RandomUnderSampler",#None,
                scaler_method=None,#scaler,
                dimensionality_reduction_method=None)#dim_red_method)
            # return

            for model in models:
                for oversampling_method in oversampling_methods:
                    logger.info(f"Running evaluation for scaler: {scaler}, dim_red_method: "
                                f"{dim_red_method}, model: {model[0]}, "
                                f"oversampling_method: "
                                f"{oversampling_method if oversample else None}")
                    output, _ = score_ml_model(X_train=X_train,
                                               y_train=y_train,
                                               X_test=X_test,
                                               y_test=y_test,
                                               model=model,
                                               oversampling_method=None,#oversampling_method,
                                               n_splits=5,
                                               target_name=target_name,
                                               scaling_method=scaler,
                                               dimensionality_reduction_method=None,#dim_red_method,
                                               oversample=oversample
                                               )
                    full_output.append(output)

    models_performance = pd.DataFrame(full_output)
    output_path = os.path.join(output_directory, f"{target_name}_models_performance_comparison.csv")
    models_performance.to_csv(output_path, index=False)

    return models_performance


def check_classification_performance(data: pd.DataFrame,
                                     target_name: str,
                                     modeling_config: Dict,
                                     model_type: str,
                                     models: list,
                                     oversampling_methods: list,
                                     oversample: bool = False,
                                     output_directory: str = "/results/model/") -> pd.DataFrame:
    """
    Creates and saves a data frame with modeling outcomes for a chosen target variable
    :param data: Pandas DataFrame
    :param target_name: name of a target variable
    :param modeling_config: JSON config file for modeling
    :param model_type: type of the model: "binary_classification" or "multiclass_classification"
    :param models: list of tuples with models names and instances for example
                   [('LogisticRegression', LogisticRegression(**kwargs)),
                   ('SGDClassifier', SGDClassifier(**kwargs))],
                   list of proposed models is saved in get_models_for_classification() function
    :param oversampling_methods: list of tuples with oversampling methods names and instances for example
                   [('SMOTE', SMOTE(**kwargs)),
                   ('ADASYN', ADASYN(**kwargs))],
                   list of proposed oversampling methods is saved in
                   get_oversampling_methods() function
    :param oversample: boolean variable if oversampling should be performed or not
    :param output_directory: directory for saving the data frame with outcomes from the modeling
    :return: pandas data frame with outcomes from the modeling
    """

    scaler_methods = modeling_config["scaler_methods"]
    #scaler_methods.append(None)
    dimensionality_reduction_methods = modeling_config["dimensionality_reduction_methods"]
    #dimensionality_reduction_methods.append(None)

    logger.info(f'---------------> {oversample}')

    if not oversample:
        oversampling_methods = [None]
    else:
        oversampling_methods = oversampling_methods  # modeling_config["oversampling_methods"]
        #oversampling_methods.append(None)

    full_output = []

    for scaler in scaler_methods:
        for dim_red_method in dimensionality_reduction_methods:
            X_train, X_test, y_train, y_test = prepare_data_for_classification(
                data=data,
                target_name=target_name,
                modeling_config=modeling_config,
                model_type=model_type,
                oversampling_method=None,#oversampling_methods[4],
                undersampling_method='RandomUnderSampler',#None,
                scaler_method=scaler,
                dimensionality_reduction_method=None)#dim_red_method)
            # return

            for model in models:
                for oversampling_method in oversampling_methods:
                    logger.info(f"Running evaluation for scaler: {scaler}, dim_red_method: "
                                f"{dim_red_method}, model: {model[0]}, "
                                f"oversampling_method: "
                                f"{oversampling_method if oversample else None}")
                    output, _ = score_ml_model(X_train=X_train,
                                               y_train=y_train,
                                               X_test=X_test,
                                               y_test=y_test,
                                               model=model,
                                               oversampling_method=None,#oversampling_method,
                                               n_splits=5,
                                               target_name=target_name,
                                               scaling_method=scaler,
                                               dimensionality_reduction_method=None,#dim_red_method,
                                               oversample=oversample
                                               )
                    full_output.append(output)

    models_performance = pd.DataFrame(full_output)
    output_path = os.path.join(output_directory, f"{target_name}_models_performance_comparison.csv")
    models_performance.to_csv(output_path, index=False)

    return models_performance


def prepare_data_for_classification(data: pd.DataFrame,
                                    target_name: str,
                                    modeling_config: Dict,
                                    model_type: str = 'binary_classification',
                                    oversampling_method: str = None,
                                    undersampling_method: str = None,
                                    scaler_method: str = None,
                                    dimensionality_reduction_method: str = None) -> (
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Transform analytical view for classification modeling purposes. The file should be in the standard wide format.
    :param data: Pandas DataFrame
    :param target_name: target variable column name
    :param modeling_config: JSON config file for modeling
    :param model_type: type of the model: "binary_classification" or "multiclass_classification"
    :param oversampling_method: oversampling method to be applied during transformation
    :param undersampling_method: undersampling method to be applied during transformation
    :param scaler_method: scaler method to be applied during transformation
    :param dimensionality_reduction_method: dimensionality reduction method to be applied during transformation
    :return: dataframes with X_train, X_test, Y_train, Y_test
    """
    #

    logger.info(f'---------------> undersampling_method {undersampling_method}')

    # if model_type == 'binary_classification':
    # target = 'target_binary_class'
    # data = drop_columns(data, ['batch_avg_value', 'target_multi_class', 'rootprocessorder'])
    # else:
    # target = 'target_multi_class'
    # data = drop_columns(data, ['batch_avg_value', 'target_binary_class', 'rootprocessorder'])

    # logger.info(f'---------------> {target_name}')
    # logger.info(f'---------------> {data.columns}')

    Y = data[target_name].copy()
    #
    X = data.drop(columns=[target_name], inplace=False, axis=1)
    # X = drop_columns(data, [target])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=modeling_config['test_size'],
                                                        random_state=modeling_config['random_state'],
                                                        shuffle=True,
                                                        stratify=Y)
    # TODO to be used once we save the final models
    # data transformation components
    # transform_components_path = pmi6_modeling_config["analytical_models_path"]
    # oversampling_name = pmi6_modeling_config["train_data_oversampler"]
    # undersampling_name = pmi6_modeling_config["train_data_undersampler"]
    # scaler_name = pmi6_modeling_config["train_test_data_scaler"]
    # dim_reduction_name = pmi6_modeling_config["train_test_data_dimensionality_reducer"]
    #
    # oversampling_pkl = os.path.join(transform_components_path, oversampling_name)
    # undersampling_pkl = os.path.join(transform_components_path, undersampling_name)
    # scaling_pkl = os.path.join(transform_components_path, scaler_name)
    # dim_reducing_pkl = os.path.join(transform_components_path, dim_reduction_name)

    # logger.info(f'!!!!!!!!!!')

    if oversampling_method:
        X_train, Y_train = over_sampling_imbalanced_dataset(X=X_train,
                                                            y=Y_train,
                                                            oversampler_pkl_file=None,
                                                            oversampling_method=oversampling_method
                                                            )

    if undersampling_method:
        X_train, Y_train = under_sampling_imbalanced_dataset(X=X_train,
                                                             y=Y_train,
                                                             undersampler_pkl_file=None,
                                                             undersampling_method=undersampling_method
                                                             )

    # return X_train, X_test, Y_train, Y_test

    if scaler_method:
        X_train, X_test = standardize_train_test_mixed_types(X_train=X_train,
                                                             X_test=X_test,
                                                             scaler_pkl_file=None,
                                                             scaler=scaler_method)
    if dimensionality_reduction_method:
        X_train, X_test = reduce_dimensionality(X_train=X_train,
                                                X_test=X_test,
                                                dim_reducer_pkl_file=None,
                                                n_components=modeling_config['dim_reduction_components'],
                                                method=dimensionality_reduction_method
                                                )

    return X_train, X_test, Y_train, Y_test


def over_sampling_imbalanced_dataset(X: pd.DataFrame,
                                     y: pd.Series,
                                     oversampler_pkl_file: str = None,
                                     oversampling_method: str = 'SMOTE',
                                     kwargs_over: Dict = {'random_state': 42}) -> (pd.DataFrame, pd.DataFrame):
    """
    Perform oversampling of the passed dataframe to balance the classes.
    Only SMOTENC is able to oversample categorical features, in all other cases categorical features are dropped!
    :param X: explaining features of the dataset
    :param y: target variable (with class indications)
    :param oversampler_pkl_file: pickle file name where oversampler will be saved
    :param oversampling_method: oversampling method: SMOTE, BorderlineSMOTE, RandomOverSampler, ADASYN, SVMSMOTE, SMOTENC
    :param kwargs_over: positional arguments for over-sampling method
    :return: dataset with balanced classes achieved with over- and under-sampling.
    """

    if not isinstance(X, pd.DataFrame):
        logger.warning(f"X={X} should be of type pd.DataFrame.")
    if not isinstance(y, pd.Series):
        logger.warning(f"y={y} should be of type pd.Series.")
    if not isinstance(oversampling_method, str):
        logger.warning(f"Oversampling_method={oversampling_method} should be of type str.")
    if not isinstance(kwargs_over, dict):
        logger.warning(f"Kwargs_over={kwargs_over} should be of type dict.")

    logger.info(f'Class size before oversampling: {Counter(y.values)}')

    # identify categorical columns (passed as objects)
    categorical_columns = X.select_dtypes(include=['object']).columns
    categorical_index = [X.columns.get_loc(each) for each in categorical_columns]

    # drop categorical columns in case of not supported categorical features
    if oversampling_method in ['SMOTE', 'BorderlineSMOTE', 'RandomOverSampler', 'ADASYN', 'SVMSMOTE']:
        logger.info('Dropping categorical variables as they are not supported by selected sampling methods.')
        X.drop(columns=categorical_columns, inplace=True)

    if oversampling_method == 'BorderlineSMOTE':
        oversample = BorderlineSMOTE(**kwargs_over)
    elif oversampling_method == 'RandomOverSampler':
        oversample = RandomOverSampler(**kwargs_over)
    elif oversampling_method == 'ADASYN':
        oversample = ADASYN(**kwargs_over)
    elif oversampling_method == 'SVMSMOTE':
        oversample = SVMSMOTE(**kwargs_over)
    elif oversampling_method == 'SMOTENC':
        oversample = SMOTENC(categorical_features=categorical_index, **kwargs_over)
    else:
        oversample = SMOTE(**kwargs_over)

    X_res, y_res = oversample.fit_resample(X.values, y.values)
    if oversampler_pkl_file:
        dump(oversample, oversampler_pkl_file, 'wb')

    logger.info(f'Class size after over-sampling with {oversampling_method}: {Counter(y_res)}')

    X_res = pd.DataFrame(data=X_res, columns=X.columns)
    y_res = pd.Series(data=y_res)
    return X_res, y_res


def under_sampling_imbalanced_dataset(X: pd.DataFrame,
                                      y: pd.Series,
                                      undersampler_pkl_file: str = None,
                                      undersampling_method: str = 'None',
                                      kwargs_under: Dict = {}) -> (pd.DataFrame, pd.DataFrame):
    """
    Perform undersampling of the passed dataframe to balance the classes.
    :param X: explaining features of the dataset
    :param y: target variable (with class indications)
    :param undersampler_pkl_file: pickle name where undersampler will be saved
    :param undersampling_method: undersampling method: None or TomekLinks
    :param kwargs_under: positional arguments for under-sampling method
    :return: dataset with balanced classes achieved with over- and under-sampling.
    """

    if not isinstance(X, pd.DataFrame):
        logger.warning(f"X={X} should be of type pd.DataFrame.")
    if not isinstance(y, pd.Series):
        logger.warning(f"y={y} should be of type pd.Series.")
    if not isinstance(undersampling_method, str):
        logger.warning(
            f"Undersampling_method={undersampling_method} should be of type str.")
    if not isinstance(kwargs_under, dict):
        logger.warning(
            f"Kwargs_under={kwargs_under} should be of type dict.")

    logger.info(f'Class size before undersampling: {Counter(y.values)}')

    # identify categorical columns (passed as objects)
    categorical_columns = X.select_dtypes(include=['object']).columns
    categorical_index = [X.columns.get_loc(each) for each in categorical_columns]

    # drop categorical columns in case of not supported categorical features
    if undersampling_method == 'TomekLinks':
        logger.info('Dropping categorical variables as they are not supported by selected sampling methods.')
        X.drop(columns=categorical_columns, inplace=True)

        undersample = TomekLinks(**kwargs_under)
        X_res, y_res = undersample.fit_resample(X.values, y.values)

        if undersampler_pkl_file:
            dump(undersample, undersampler_pkl_file, 'wb')

        logger.info(f'Class size after under-sampling with {undersampling_method}: {Counter(y_res)}')
        X_res = pd.DataFrame(data=X_res, columns=X.columns)
        y_res = pd.Series(data=y_res)
    elif undersampling_method == 'RandomUnderSampler':
        #logger.info('Dropping categorical variables as they are not supported by selected sampling methods.')
        #X.drop(columns=categorical_columns, inplace=True)

        logger.info('RandomUnderSampler')

        undersample = RandomUnderSampler(**kwargs_under)
        X_res, y_res = undersample.fit_resample(X.values, y.values)

        if undersampler_pkl_file:
            dump(undersample, undersampler_pkl_file, 'wb')

        logger.info(f'Class size after under-sampling with {undersampling_method}: {Counter(y_res)}')
        X_res = pd.DataFrame(data=X_res, columns=X.columns)
        y_res = pd.Series(data=y_res)
    else:
        X_res = X
        y_res = y
    return X_res, y_res


def standardize_train_test_mixed_types(X_train: pd.DataFrame,
                                       X_test: pd.DataFrame,
                                       scaler_pkl_file: str = None,
                                       scaler: str = "StandardScaler") -> (pd.DataFrame,
                                                                           pd.DataFrame):
    """
    Function standardizes numerical columns of X_train and X_test datasets.
    :param X_train: explaining features of train set
    :param X_test: explaining features of test set
    :param scaler_pkl_file: pickle file name where scaler is to be saved
    :param scaler: type of standardized used: "StandardScaler", "RobustScaler", "Normalizer"
    :return: tuple of standardized X_train and X_test
    """
    categorical_train = X_train.select_dtypes("object")
    categorical_test = X_test.select_dtypes("object")

    numerical_train = X_train.select_dtypes("number")
    numerical_test = X_test.select_dtypes("number")

    if scaler == "RobustScaler":
        scaler_model = RobustScaler()
    elif scaler == "Normalizer":
        scaler_model = Normalizer()
    else:
        scaler_model = StandardScaler()

    scaler_model.fit(numerical_train)

    if scaler_pkl_file:
        dump(scaler_model, open(scaler_pkl_file, 'wb'))

    numerical_train_scaled = scaler_model.transform(numerical_train)
    numerical_test_scaled = scaler_model.transform(numerical_test)

    # converting scaled numerical data to data frame
    numerical_train_scaled_df = pd.DataFrame(data=numerical_train_scaled[0:, 0:],
                                             index=categorical_train.index,
                                             columns=[numerical_train.columns[i] for i in
                                                      range(numerical_train_scaled.shape[1])])
    numerical_test_scaled_df = pd.DataFrame(data=numerical_test_scaled[0:, 0:],
                                            index=categorical_test.index,
                                            columns=[numerical_test.columns[i] for i in
                                                     range(numerical_test_scaled.shape[1])])

    # joining categorical data  with scaled numerical data
    X_train_scaled = pd.concat([categorical_train, numerical_train_scaled_df], axis=1)
    X_test_scaled = pd.concat([categorical_test, numerical_test_scaled_df], axis=1)

    return X_train_scaled, X_test_scaled


def reduce_dimensionality(X_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          dim_reducer_pkl_file: str = None,
                          n_components: int = 3,
                          method: str = "PCA",
                          random_state: int = 42) -> (pd.DataFrame, pd.DataFrame):
    """
    Function reduces the dimensionality of independent features set for train and test sets to
    number of dimensions chosen by the user.
    :param X_train: independent features from train set
    :param X_test: independent features from test set
    :param dim_reducer_pkl_file: pickle file where dimmensionality reducer will be saved
    :param n_components: number of dimensions to which we want to reduce the set of independent
    features
    :param method: method of dimensionality reduction used, user can choose among "PCA",
    "TruncatedSVD", "FastICA". The default method is PCA.
    :param random_state: random state used in dimensionality reduction method
    :return: tuple of independent features sets for train and test data after dimensionality
    reduction
    """
    if method == "TruncatedSVD":
        model = TruncatedSVD(n_components=n_components, random_state=random_state)
    elif method == "FastICA":
        model = FastICA(n_components=n_components, random_state=random_state)
    else:
        model = PCA(n_components=n_components, random_state=random_state)

    X_train_reduced_dim = model.fit_transform(X_train)

    if dim_reducer_pkl_file:
        dump(model, dim_reducer_pkl_file, 'wb')

    X_test_reduced_dim = model.transform(X_test)

    X_train_reduced_dim = pd.DataFrame(data=X_train_reduced_dim[0:, 0:],
                                       index=X_train.index,
                                       columns=[f'Component_{i + 1}' for i in range(n_components)])

    X_test_reduced_dim = pd.DataFrame(data=X_test_reduced_dim[0:, 0:],
                                      index=X_test.index,
                                      columns=[f'Component_{i + 1}' for i in range(n_components)])

    return X_train_reduced_dim, X_test_reduced_dim


def score_ml_model(X_train: pd.DataFrame,
                   y_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame,
                   model: tuple,
                   oversampling_method: tuple or None,
                   target_name: str,
                   scaling_method: str,
                   dimensionality_reduction_method: str,
                   n_splits: int = 5,
                   oversample: bool = True) -> (list, list):
    """
    Measures the performance of the given configuration of model and oversampling method with
    F1, recall and precision metrics on Cross-Validation (train set) and on the test set
    :param X_train: pandas DataFrame with train independent variables
    :param y_train: pandas DataFrame with train dependent variable
    :param X_test: pandas DataFrame with test independent variables
    :param y_test: pandas DataFrame with test dependent variable
    :param model: tuple with model name and model instance f.e.
                  ('LogisticRegression', LogisticRegression(**kwargs))
    :param oversampling_method: tuple with oversampling method name and instance f.e.
                                ('SMOTE', SMOTE(**kwargs))
    :param target_name: name of target variable
    :param scaling_method: name of the scaling method f.e. "StandardScaler"
    :param dimensionality_reduction_method: name of the dimensionality reduction method f.e. "PCA"
    :param n_splits: number of splits for KFold
    :param oversample: boolean variable if oversampling should be performed or not
    :return: a) output dictionary with details about used scaling, dimensionality reduction and
             oversampling method, model and its performance measured by F1, precision and recall
             metrics both for average value from Cross-Validation and test results
             b) results list with F1, precision and recall values for each of KFold
             Cross-Validation split
    """
    results = []

    try:
        kf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
        if oversample:
            pipeline = make_pipeline(oversampling_method[1], model[1])
        else:
            pipeline = make_pipeline(model[1])
            oversampling_method = None

        cv_results = cross_validate(pipeline, X_train, y_train, scoring=["recall", "precision", "f1", "accuracy"], cv=kf)
        results.append(cv_results)




        mod = pipeline.fit(X_train, y_train)
        y_pred = mod.predict(X_test)

        f1_test_score = f1_score(y_test, y_pred)
        precision_test_score = precision_score(y_test, y_pred)
        recall_test_score = recall_score(y_test, y_pred)
        accuracy_test_score = accuracy_score(y_test, y_pred)

        # https://deepai.org/machine-learning-glossary-and-terms/f-score
        beta_squared = 2^2
        # recall is 2x more important than precission, we want to make sure we select as many ketosis cows as possible
        f2_score_test = (1 + beta_squared) * (precision_test_score * recall_test_score) / ((beta_squared * precision_test_score) + recall_test_score)

        output = {"target_name": target_name,
                  "scaling_method": scaling_method,
                  "dimensionality_reduction_method": dimensionality_reduction_method,
                  "model_name": model[0],
                  "oversampling_method_name": oversampling_method[0] if oversampling_method else None,
                  "CV_mean_F1": cv_results['test_f1'].mean(),
                  "CV_std_F1": cv_results['test_f1'].std(),
                  "CV_mean_precision": cv_results['test_precision'].mean(),
                  "CV_std_precision": cv_results['test_precision'].std(),
                  "CV_mean_recall": cv_results['test_recall'].mean(),
                  "CV_std_recall": cv_results['test_recall'].std(),
                  "CV_mean_accuracy": cv_results['test_accuracy'].mean(),
                  "CV_std_accuracy": cv_results['test_accuracy'].std(),
                  "test_F1": f1_test_score,
                  "test_precision": precision_test_score,
                  "test_recall": recall_test_score,
                  "test_accuracy": accuracy_test_score,
                  "f2_score_test": f2_score_test
                  }

    except:
        output = {"target_name": target_name,
                  "scaling_method": scaling_method,
                  "dimensionality_reduction_method": dimensionality_reduction_method,
                  "model_name": model[0],
                  "oversampling_method_name": oversampling_method[0] if oversampling_method else None,
                  "CV_mean_F1": np.nan,
                  "CV_std_F1": np.nan,
                  "CV_mean_precision": np.nan,
                  "CV_std_precision": np.nan,
                  "CV_mean_recall": np.nan,
                  "CV_std_recall": np.nan,
                  "CV_mean_accuracy": np.nan,
                  "CV_std_accuracy": np.nan,
                  "test_F1": np.nan,
                  "test_precision": np.nan,
                  "test_recall": np.nan,
                  "test_accuracy": np.nan
                  }

    return output, results


def select_models(df: pd.DataFrame,
                  min_threshold: float,
                  max_relative_dif: float,
                  evaluation_results_csv_path: str = None):
    """
    Selects models from evaluated models according to the established logic.
    :param df: data frame with performance metrics
    :param evaluation_results_csv_path: the csv path to the evaluation results
    :param min_threshold: min acceptable F1, recall and precision from CV
    :param max_relative_dif: max acceptable relative difference between CV_mean_F1 and test_F1,
                             CV_mean_recall and test_recall, CV_mean_precision and test_precision
    :return: the data frame which contains only selected models
    """
    #logger = choose_logger_type(logger_type)
    if evaluation_results_csv_path:
        df = pd.read_csv(evaluation_results_csv_path)

    # CV_mean_F1 & CV_mean_precision & CV_mean_recall are higher than min_threshold
    cv_mask = (df["CV_mean_F1"] > min_threshold) & (df["CV_mean_precision"] > min_threshold) & (df["CV_mean_recall"] > min_threshold)
    df = df[cv_mask]

    # test_F1 & test_precision & test_recall are higher than min_threshold
    test_mask = (df["test_F1"] > min_threshold) & (df["test_precision"] > min_threshold) & (df["test_recall"] > min_threshold)
    df = df[test_mask]

    # test_recall <= CV_mean_recall * (1 + max_relative_dif)
    test_recall_mask = df[['CV_mean_recall', 'test_recall']].apply(
        lambda x: x['test_recall'] >= x['CV_mean_recall'] * (1 - max_relative_dif),
        axis=1
    )
    df = df[test_recall_mask]

    # test_F1 <= CV_mean_F1 * (1 + max_relative_dif)
    test_f1_mask = df[['CV_mean_F1', 'test_F1']].apply(
        lambda x: x['test_F1'] >= x['CV_mean_F1'] * (1 - max_relative_dif),
        axis=1
    )
    df = df[test_f1_mask]

    # test_precision <= CV_mean_precision * (1 + max_relative_dif)
    test_precision_mask = df[['CV_mean_precision', 'test_precision']].apply(
        lambda x: x['test_precision'] >= x['CV_mean_precision'] * (1 - max_relative_dif),
        axis=1
    )
    df = df[test_precision_mask]

    if df.empty:
        logger.warning(
            f'0 models are selected, please revise thresholds: min_threshold={min_threshold} and '
            f'max_relative_dif={max_relative_dif}.')
    else:
        logger.info(f'There are N={df.shape[0]} selected models.')

    return df
