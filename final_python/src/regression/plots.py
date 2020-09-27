from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import RegressionResults


def plot_residuals_vs_fitted(
        target: pd.Series or np.ndarray,
        prediction: pd.Series or np.ndarray,
        figsize: Tuple[int, int] = (10, 7)) -> None:
    """
    Plots residuals vs values fitted by the model and actual values vs fitted values.
    :param target: vector with dependent variable
    :param prediction: vector with values predicted by the model
    :param figsize: the size of figure
    """

    residuals = target - prediction
    res_scores = (r'$mean={:.2f}$' + '\n' + r'$std={:.2f}$').format(
        np.mean(residuals),
        np.std(residuals)
    )

    plt.figure(figsize=figsize)
    ax = sns.residplot(prediction, residuals,
                       lowess=True,
                       scatter_kws={'alpha': 0.5},
                       line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    ax.set_title('Residuals vs Fitted')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor='none', linewidth=0)
    ax.legend([extra], [res_scores], loc='upper left')

    plt.show()


def plot_actuals_vs_fitted(
        target: pd.Series or np.ndarray,
        prediction: pd.Series or np.ndarray,
        r2_score: float = None,
        figsize: Tuple[int, int] = (10, 7)) -> None:
    """
    Plots residuals vs values fitted by the model and actual values vs fitted values.

    :param target: vector with dependent variable
    :param prediction: vector with values predicted by the model
    :param r2_score: R-squared statistic for model
    :param figsize: the size of figure
    """

    plt.figure(figsize=figsize)
    ax = sns.scatterplot(target, prediction)
    ax.plot([target.min(), target.max()],
            [target.min(), target.max()],
            '--r', linewidth=2)

    ax.set_title('Actual vs Fitted')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Actual values')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if r2_score:
        scores = r'$R^2={:.2f}$'.format(
            r2_score
        )
        extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor='none', linewidth=0)
        ax.legend([extra], [scores], loc='upper left')

    plt.show()


def plot_standarized_residuals_vs_fitted(fitted_model: RegressionResults) -> None:
    """
    Plots standarized residuals vs values fitted by the OLS model.
    :param fitted_model: instance of fitted model
    """

    fitted = fitted_model.fittedvalues
    student_residuals = fitted_model.get_influence().resid_studentized_internal
    sqrt_student_residuals = pd.Series(np.sqrt(np.abs(student_residuals)))
    sqrt_student_residuals.index = fitted_model.resid.index
    smoothed = lowess(sqrt_student_residuals, fitted)

    fig, ax = plt.subplots()
    ax.scatter(fitted, sqrt_student_residuals, edgecolors='k', facecolors='none')
    ax.plot(smoothed[:, 0], smoothed[:, 1], color='r')
    ax.set_ylabel('$\sqrt{|Studentized \ Residuals|}$')
    ax.set_xlabel('Fitted Values')
    ax.set_title('Scale-Location')
    ax.set_ylim(0, max(sqrt_student_residuals) + 0.1)
    plt.show()


def plot_independent_variables_vs_other(
        data: pd.DataFrame or np.ndarray,
        column_name: str,
        variable: pd.Series or np.ndarray,
        y_label: str = None,
        axhline: bool = False) -> None:
    """
    Plots chosen independent variable vs other feature (f.g. residuals, target variable)
    :param data: DataFrame with independent variables
    :param column_name: name of the column from independent variables to plot
    :param variable: other variable to plot against
    :param y_label: name of the other variable
    :param axhline: if horizontal line on 0 should be visible
    """
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data[column_name], variable)
    plt.ylabel(y_label)
    if axhline:
        plt.axhline(0, ls="--")
    plt.show()
