import pandas as pd


class Color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def test_hypothesis(significance_level: float,
                    p_value: float or int,
                    null_hypothesis: str = None,
                    print_outcome: bool = True,
                    reject_null_hypothesis: bool = False) -> bool:
    """
    Tests if once can reject the null hypothesis of statistical test
    :param significance_level: level of significance (the probability of rejecting the null
                               hypothesis given that it is true)
    :param p_value: p_value for the statistical test
    :param null_hypothesis: null hypothesis of statistical test
    :param print_outcome: if the sentence about rejecting/not rejecting null hypothesis should be
                          printed
    :param reject_null_hypothesis: if we want to reject null hypothesis to fulfill the
                                   assumption
    """
    if reject_null_hypothesis:
        if p_value < significance_level:
            if print_outcome:
                print("We reject the null hypothesis that " + f"{null_hypothesis} " + Color.GREEN +
                      "=> Assumption satisfied" + Color.END + "\n")
                return True
            return True
        else:
            if print_outcome:
                print("We can't reject the null hypothesis that " + f"{null_hypothesis} " +
                      Color.RED + "=> Assumption not satisfied" + Color.END + "\n")
                return False
            return False
    else:
        if p_value < significance_level:
            if print_outcome:
                print("We reject the null hypothesis that " + f"{null_hypothesis} " + Color.RED +
                      "=> Assumption not satisfied" + Color.END + "\n")
                return False
            return False
        else:
            if print_outcome:
                print("We can't reject the null hypothesis that " + f"{null_hypothesis} " +
                      Color.GREEN + "=> Assumption satisfied" + Color.END + "\n")
                return True
            return True


def test_correlation(p_value: float or int,
                     correlation: float or int,
                     variable_name: str,
                     significance_level: float,
                     reject_null_hypothesis: bool = False) -> None:
    """
    Tests pearson's correlation between two variables.
    :param p_value: value of p_value for correlation test
    :param correlation: value of the correlation
    :param variable_name: name of the variable for which we check the correlation
    :param significance_level: level of significance (the probability of rejecting the null
                               hypothesis given that it is true)
    :param reject_null_hypothesis: if we want to reject null hypothesis to fulfill the
                                   assumption
    """

    if reject_null_hypothesis:
        if p_value < significance_level:
            print(f"Variable {variable_name}: Pearson's correlation: " + Color.GREEN
                  + f"{correlation:.4f}" + Color.END
                  )
        else:
            print(f"Variable {variable_name}: Pearson's correlation: " +
                  Color.RED + f"{correlation:.4f}" + Color.END)
    else:
        if p_value < significance_level:
            print(f"Variable {variable_name}: Pearson's correlation: " + Color.RED
                  + f"{correlation:.4f}" + Color.END
                  )
        else:
            print(f"Variable {variable_name}: Pearson's correlation: " +
                  Color.GREEN + f"{correlation:.4f}" + Color.END)


def create_df_with_most_and_least_correlated_features(
        corr_df: pd.DataFrame,
        corr_column: str,
        max_variables: int,
        return_most_correlated: bool = False,
        return_least_correlated: bool = False) -> pd.DataFrame:
    """
    Creates data frame with max_variables/2 most correlated and max_variables/2 least correlated
    features
    :param corr_df: table with correlations
    :param corr_column: name of column with correlations
    :param max_variables: maximum variables in the output data frame
    :param return_most_correlated: boolean variable if we want to create df with only most
                                   correlated variables
    :param return_least_correlated: boolean variable if we want to create df with only least
                                    correlated variables
    Return: data frame with most or/and least correlated features
    """

    sorted_df = corr_df.abs().sort_values(by=[corr_column], ascending=False)

    most_correlated = corr_df.loc[sorted_df.index].head(int(max_variables / 2))
    least_correlated = corr_df.loc[sorted_df.index].tail(int(max_variables / 2))

    if return_most_correlated:
        return most_correlated
    elif return_least_correlated:
        return least_correlated
    else:
        return pd.concat([most_correlated, least_correlated])


def check_fulfill_ratio(true_fulfill_ratio: float,
                        min_fulfill_ratio: float,
                        print_outcome: bool = True) -> bool:
    """
    Checks if the percentage of tests that fulfilled the assumption is greater than
    min_fulfill_ratio
    :param true_fulfill_ratio: real percentage of tests that fulfilled the assumption
    :param min_fulfill_ratio: minimal percentage of statistical tests that fulfilled the assumption
    :param print_outcome: if the sentence about fulfilling/not fulfilling the assumption should be
                          printed
    """
    if true_fulfill_ratio >= min_fulfill_ratio:
        if print_outcome:
            print("Based on the performed statistical tests, " + Color.GREEN + "the assumption is "
                                                                               "satisfied.\n" +
                  Color.END)
        return True
    else:
        if print_outcome:
            print("Based on the performed statistical tests, " + Color.RED + "the assumption is "
                                                                             "not satisfied.\n" +
                  Color.END)
        return False
