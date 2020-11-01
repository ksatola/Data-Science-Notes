from numpy import mean
from numpy import std
from numpy import percentile
import pandas as pd


def identify_outliers(*, data: pd.Series, cut_off_limit: int, method: str = "std") -> list:
    """
    Removes outliers from a Pandas series defined as values exceeding absolute values greater than
    the cut_off standard deviations.
    :param data: Pandas series
    :param cut_off_limit: number of times of standard deviation or IQR for outlier boundary selection
    :param method: can be "std" for normally distributed data or "iqr" for non-Gaussian distribution
    :return: list of indexes in the original Pandas series identified as outliers
    """
    """
    
    method can be "std" for normally distributed data or "iqr" for non-Gaussian distribution
    """

    data_vals = data.values.ravel()
    # print(data_vals)

    if method == "iqr":
        # Calculate interquartile range
        q25, q75 = percentile(data, [25, 75])
        iqr = q75 - q25

        # Calculate the outlier cutoff
        cut_off = iqr * cut_off_limit
        lower, upper = q25 - cut_off, q75 + cut_off

        print(f"Percentiles: 25th={q25:.3f}, 75th={q75:.3f}, "
              f"IQR={iqr:.3f}, cut_off: {cut_off:.03f}, "
              f"lower_limit: {lower:.03f}, upper_limit: {upper:.03f}")

    else:
        # Calculate summary statistics
        data_mean, data_std = mean(data_vals), std(data_vals)

        # Identify outliers
        cut_off = data_std * cut_off_limit
        lower = data_mean - cut_off
        upper = data_mean + cut_off

        print(f"Mean: {data_mean:.03f}, cut_off: {cut_off:.03f}, "
              f"lower_limit: {lower:.03f}, upper_limit: {upper:.03f}")

    outlier_indexes = []
    for index, value in data.iteritems():
        if value < lower or value > upper:
            outlier_indexes.append(index)
            # print(value)

    return outlier_indexes
