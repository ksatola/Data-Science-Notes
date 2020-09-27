from numbers import Number

import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
import statsmodels.stats.diagnostic as stats_diag
from .helpers import (
    Color,
    test_hypothesis,
    test_correlation,
    create_df_with_most_and_least_correlated_features,
    check_fulfill_ratio
)
from matplotlib import pyplot as plt
from .plots import (
    plot_residuals_vs_fitted,
    plot_actuals_vs_fitted,
    plot_standarized_residuals_vs_fitted,
    plot_independent_variables_vs_other
)
from scipy import stats
from scipy.stats.stats import pearsonr
from statsmodels import api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import (
    durbin_watson,
    omni_normtest
)


class CheckOLSAssumptions:
    """
    Class for checking following Ordinary Least Squares Regression assumptions:
    1. Regression model is linear in the coefficients and the error term.
    2. The error term has mean of 0.
    3. All independent variables are uncorrelated with the error term.
    4. Observations of the error term are uncorrelated with each other.
    5. The error term has a constant variance.
    6. No independent variable is a perfect linear function of other explanatory variables.
    7. The error term is normally distributed.
    """

    def __init__(self,
                 data: pd.DataFrame or np.ndarray,
                 target: pd.Series or np.ndarray,
                 alpha: float = 0.05,
                 vif_threshold: float or int = 5,
                 max_independent_variables_to_show: int = 10,
                 min_fulfill_ratio: float = 0.5,
                 silent_mode: bool = False):
        """
        :param data: pandas DataFrame or numpy array with independent variables used for training
                     (without Intercept variable)
        :param target: pandas Series or numpy array with dependent variable used for training
        :param alpha: significance level for statistical tests (the probability of rejecting the
                      null hypothesis when it is true)
        :param vif_threshold: threshold for Variance Inflation Factor
        :param max_independent_variables_to_show: maximum number of independent variables to be
                                                  shown on the plots/statistics
        :param min_fulfill_ratio: minimal percentage of statistical tests for which the assumption
                                  has to be fulfilled to be found as fulfilled overall
        :param silent_mode: if True: the method returns only True/False if the assumption is
                            fulfilled/not fulfilled (based on the detailed guidelines given in
                            every method separately),
                            if False: the method prints the whole description and statistics and
                            returns True/False as above
        """
        self.validate_input_parameters(data,
                                       target,
                                       alpha,
                                       vif_threshold,
                                       max_independent_variables_to_show,
                                       min_fulfill_ratio,
                                       silent_mode)

        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        if isinstance(target, np.ndarray):
            target = pd.Series(target, name="target variable")

        self.data = data.assign(Intercept=1)
        self.target = target
        self.model = sm.OLS(target, data, hasconst=True)
        self.results = self.model.fit()
        self.r2_score = self.results.rsquared
        self.prediction = self.results.predict(data)
        self.residuals = self.target - self.prediction
        self.corr_with_target = pd.DataFrame(
            [pearsonr(self.data[c], self.target) for c in self.data.columns.drop("Intercept")],
            columns=["target_corr", "target_p_value"],
            index=self.data.columns.drop("Intercept")
        )
        self.corr_with_residuals = pd.DataFrame(
            [pearsonr(self.data[col], self.residuals) for col in
             self.data.columns.drop("Intercept")],
            columns=["residuals_corr", "residuals_p_value"],
            index=self.data.columns.drop("Intercept")
        )
        self.max_iv_show = max_independent_variables_to_show
        self.alpha = alpha
        self.vif_threshold = vif_threshold
        self.silent_mode = silent_mode
        self.min_fulfill_ratio = min_fulfill_ratio

    @staticmethod
    def validate_input_parameters(data: pd.DataFrame or np.ndarray,
                                  target: pd.Series or np.ndarray,
                                  alpha: float,
                                  vif_threshold: float or int,
                                  max_independent_variables_to_show: int,
                                  min_fulfill_ratio: float,
                                  silent_mode: bool):
        """Validates if class' parameters are correctly specified."""

        if isinstance(data, np.ndarray):
            if not np.vectorize(lambda x: isinstance(x, Number))(data).all():
                raise ValueError("All independent variables should be numeric in linear "
                                 "regression, but there appear non-numeric data.")
        elif isinstance(data, pd.DataFrame):
            if not data.applymap(lambda x: isinstance(x, Number)).all(axis=None):
                raise ValueError("All independent variables should be numeric in linear "
                                 "regression, but there appear non-numeric data.")
        else:
            raise ValueError("Variable data should be numpy array or pandas DataFrame.")

        if isinstance(target, np.ndarray):
            if not np.vectorize(lambda x: isinstance(x, Number))(target).all():
                raise ValueError("Target variable should be numeric in linear regression, "
                                 "but there appear non-numeric data.")
        elif isinstance(target, pd.Series):
            if not target.map(lambda x: isinstance(x, Number)).all(axis=None):
                raise ValueError("Target variable should be numeric in linear regression, "
                                 "but there appear non-numeric data.")
        else:
            raise ValueError("Variable data should be numpy array or pandas Series.")

        if data.shape[0] != target.shape[0]:
            raise ValueError(f"The first dimension of data ({data.shape[0]}) is not equal to "
                             f"the first dimension of target variable ({target.shape[0]}).")

        if not isinstance(alpha, float) or not 0 < alpha < 1:
            raise ValueError(
                f"alpha={alpha} is a significance level which is a probability of rejecting null "
                f"hypothesis so it must be a positive number between [0,1].")

        if not isinstance(vif_threshold, (int, float)) or vif_threshold <= 0:
            raise ValueError(f"vif_threshold={vif_threshold} is threshold for Variance "
                             f"Inflation Factor so it must be a positive number.")

        if not isinstance(max_independent_variables_to_show, int) or (
                max_independent_variables_to_show <= 0):
            raise ValueError(
                f"max_independent_variables_to_show={max_independent_variables_to_show} should "
                f"be a positive integer as it is maximum number of independent variables to plot.")

        if not isinstance(min_fulfill_ratio, float) or not 0 < min_fulfill_ratio <= 1:
            raise ValueError(f"min_fulfill_ratio={min_fulfill_ratio} is minimal percentage of "
                             f"statistical tests for which the assumption is fulfilled so it "
                             f"should be positive float from [0,1] interval.")

        if not isinstance(silent_mode, bool):
            raise ValueError(
                f"silent_mode={silent_mode} should be a boolean variable as it lets you decide if "
                f"you would like to see only True/False for fulfilled/not fulfilled assumption "
                f"or the whole description of assumption."
            )

    def summarize_model(self):
        """Summarizes the model with basic statistics."""
        if not self.silent_mode:
            print("Below you can find the summary of the model: \n")
            print(self.results.summary(), "\n")
        else:
            pass

    def check_linearity(self) -> bool:
        """
        Checks if there exists linear dependency between the independent variables and dependent
        variable.
        If:
         - silent_mode = True, method returns:
                                              a) True (which means that the assumption is
                                                 fulfilled) if there exists at least one
                                                 independent variable that has statistically
                                                 significant linear correlation with target
                                                 variable
                                              b) False (which means that the assumption is not
                                                 fulfilled) if there isn't any independent variable
                                                 that has statistically significant linear
                                                 correlation with target variable
         - silent_mode = False, method returns True/False as above and shows the statistics, plots
         and descriptions which are helpful in assessing the fulfilment of assumption
        """

        correlation_df = create_df_with_most_and_least_correlated_features(
            corr_df=self.corr_with_target,
            corr_column="target_corr",
            max_variables=self.max_iv_show
        )

        true_counts = 0
        if not correlation_df[correlation_df['target_p_value'] < self.alpha].empty:
            true_counts = true_counts + 1

        true_ratio = true_counts

        if not self.silent_mode:
            print(Color.BOLD + "Assumption 1. Regression model is linear in the coefficients and "
                               "the error term. \n" + Color.END)

            print("This assumption affects on: \n",
                  "- prediction \n",
                  "- interpretation \n")

            print("Residuals vs Fitted and Actual vs Fitted plots \n")
            print(Color.BOLD + "1. Check Residuals vs Fitted plot. \n\n" + Color.END +
                  "The points on Residuals vs Fitted plot should be randomly dispersed around 0, "
                  "so in the perfect linear model, the red line should be horizontal line "
                  "y = 0.\n\n"
                  "An unbiased model has residuals that are randomly scattered around zero. "
                  "Non-random residual patterns indicate a bad fit despite a high R^2. This type "
                  "of specification bias occurs when the model is missing significant independent "
                  "variables, polynomial terms, and interaction terms. \n\n")

            plot_residuals_vs_fitted(target=self.target,
                                     prediction=self.prediction,
                                     )

            print(Color.BOLD + "2. Check Actual vs Fitted plot.\n\n" + Color.END +
                  "The points on Actual vs Fitted plot should be randomly dispersed around the "
                  "red line.\n\n"
                  "The situation when the points are not randomly dispersed on any of those two "
                  "plots, gives the first signal that some of the models' assumptions can be "
                  "violated. \n")

            plot_actuals_vs_fitted(target=self.target,
                                   prediction=self.prediction,
                                   r2_score=self.r2_score)

            print(Color.BOLD + "3. Check if there exists linear or non-linear relationship "
                               "between the independent variables and target variable. \n\n" +
                  Color.END +
                  "If there exists linear relationship:\n"
                  "- the correlation coefficient is listed in " + Color.GREEN + "green color, " +
                  Color.END + "otherwise in " + Color.RED + "red color.\n" + Color.END +
                  "- target variable can be explained by including linear terms of independent "
                  "variables in the model. \n")

            if len(self.data.columns) > self.max_iv_show:
                print(f"Due to the fact that the dataset contains more than {self.max_iv_show} "
                      f"independent variables, there will be showed {int(self.max_iv_show / 2)} "
                      f"most and {int(self.max_iv_show / 2)} least correlated ones.\n"
                      "- If you want to see more independent variables, please change the class' "
                      "parameter: max_independent_variables_to_show to the required number.\n")

                for index, row in correlation_df.iterrows():
                    if index == correlation_df.index[0]:
                        print("The most correlated independent variables: \n")
                    elif index == correlation_df.index[int(self.max_iv_show / 2)]:
                        print("The least correlated independent variables: \n")

                    test_correlation(p_value=row["target_p_value"],
                                     correlation=row["target_corr"],
                                     variable_name=index,
                                     significance_level=self.alpha,
                                     reject_null_hypothesis=True)

                    plot_independent_variables_vs_other(data=self.data,
                                                        column_name=index,
                                                        variable=self.target,
                                                        y_label=self.target.name,
                                                        axhline=False)

            else:
                for index, row in self.corr_with_target.sort_values(by='target_corr').iterrows():
                    test_correlation(p_value=row["target_p_value"],
                                     correlation=row["target_corr"],
                                     variable_name=index,
                                     significance_level=self.alpha,
                                     reject_null_hypothesis=True)

                    plot_independent_variables_vs_other(data=self.data,
                                                        column_name=index,
                                                        variable=self.target,
                                                        y_label="target variable",
                                                        axhline=False)
            check_fulfill_ratio(true_fulfill_ratio=true_ratio,
                                min_fulfill_ratio=self.min_fulfill_ratio
                                )

        return check_fulfill_ratio(true_fulfill_ratio=true_ratio,
                                   min_fulfill_ratio=self.min_fulfill_ratio,
                                   print_outcome=False)

    def check_error_term_zero_mean(self) -> bool:
        """
        Checks if the mean value of error term is 0 by one sample t-test.
        If:
         - silent_mode = True, method returns:
                                              a) True (which means that the assumption is
                                                 fulfilled) if zero hypothesis of one sample
                                                 t-test for mean is not rejected
                                              b) False (which means that the assumption is not
                                                 fulfilled) iif zero hypothesis of one sample
                                                 t-test for mean is rejected
         - silent_mode = False, method returns True/False as above and shows additional statistics,
         descriptions which are helpful in assessing the fulfilment of assumption
        """

        t_test = stats.ttest_1samp(self.residuals, popmean=0)

        true_counts = 0
        true_counts = true_counts + test_hypothesis(significance_level=self.alpha,
                                                    p_value=t_test[1],
                                                    print_outcome=False)
        true_ratio = true_counts

        if not self.silent_mode:
            print(Color.BOLD + "Assumption 2. The error term has mean of 0. \n" + Color.END)

            print("This assumption affects on: \n",
                  "- prediction \n",
                  "- interpretation \n")

            print("If the average error is +7, this non-zero error indicates that our model "
                  "systematically underpredicts the observed values. Statisticians refer to "
                  "systematic error like this as bias, and it signifies that our model is "
                  "inadequate because it is not correct on average.\n")

            print(f"Mean value of the error term is {np.mean(self.residuals): .4f} \n")

            print(Color.BOLD + "Based on one sample t-test for mean: \n" + Color.END)

            test_hypothesis(significance_level=self.alpha,
                            p_value=t_test[1],
                            null_hypothesis="mean of the error term is equal to 0.")

            check_fulfill_ratio(true_fulfill_ratio=true_ratio,
                                min_fulfill_ratio=self.min_fulfill_ratio
                                )

        return check_fulfill_ratio(true_fulfill_ratio=true_ratio,
                                   min_fulfill_ratio=self.min_fulfill_ratio,
                                   print_outcome=False)

    def check_independent_variables_vs_error_correlation(self) -> bool:
        """
        Checks correlation between independent variables and the error term in two ways:
            - linear correlation by Pearson linear correlation test,
            - non-linear correlation on scatter plots.
        If:
         - silent_mode = True, method returns:
                                              a) True (which means that the assumption is
                                                 fulfilled) if there isn't any independent variable
                                                 that has statistically significant linear
                                                 correlation with the error term
                                              b) False (which means that the assumption is not
                                                 fulfilled) if there exists at least one
                                                 independent variable that has statistically
                                                 significant linear correlation with the error term
         - silent_mode = False, method returns True/False as above and shows additional statistics,
         descriptions which are helpful in assessing the fulfilment of assumption
        """

        correlation_df = create_df_with_most_and_least_correlated_features(
            corr_df=self.corr_with_residuals,
            corr_column="residuals_corr",
            max_variables=2 * self.max_iv_show,
            return_most_correlated=True
        )

        true_counts = 0
        if correlation_df[correlation_df['residuals_p_value'] < self.alpha].empty:
            true_counts = true_counts + 1

        true_ratio = true_counts

        if not self.silent_mode:
            print(Color.BOLD + "Assumption 3. All independent variables are uncorrelated with the "
                               "error term." + Color.END, "\n")

            print("This assumption affects on: \n",
                  "- prediction \n",
                  "- interpretation \n")

            print("If an independent variable is correlated with the error term, we can use the "
                  "independent variable to predict the error term, which violates the notion that "
                  "the error term represents unpredictable random error. Violations of this "
                  "assumption can occur because there is simultaneity between the independent and "
                  "dependent variables, omitted variable bias, incorrectly modeled curvature, or "
                  "measurement error in the independent variables.\n")

            print(Color.BOLD + "Check if there exists linear or non-linear relationship of any "
                               "independent variable with the error term.\n\n" + Color.END +
                  "If there exists linear relationship:\n"
                  "- the correlation coefficient is listed in " + Color.RED + "red color, " +
                  Color.END + "otherwise in " + Color.GREEN + "green color.\n" + Color.END)

            if len(self.data.columns) > self.max_iv_show:
                print(f"Due to the fact that the dataset contains more than {self.max_iv_show} "
                      f"independent variables, there will be showed {self.max_iv_show} most "
                      f"correlated variables (if there exists significant linear correlation "
                      f"the Pearson's correlation is listed in " + Color.RED + "red colour " +
                      Color.END + "as it is a signal of not fulfilling the assumption).\n"
                                  "- If you want to see more independent variables, please change "
                                  "the class' parameter: max_independent_variables_to_show to the "
                                  "required number.\n")

                for index, row in correlation_df.iterrows():
                    if index == correlation_df.index[0]:
                        print("The most correlated independent variables: \n")

                    test_correlation(p_value=row["residuals_p_value"],
                                     correlation=row["residuals_corr"],
                                     variable_name=index,
                                     significance_level=self.alpha,
                                     reject_null_hypothesis=False)

                    plot_independent_variables_vs_other(data=self.data,
                                                        column_name=index,
                                                        variable=self.residuals,
                                                        y_label="residuals",
                                                        axhline=True)

            else:
                for index, row in self.corr_with_residuals.sort_values(
                        by='residuals_corr').iterrows():
                    test_correlation(p_value=row["residuals_p_value"],
                                     correlation=row["residuals_corr"],
                                     variable_name=index,
                                     significance_level=self.alpha,
                                     reject_null_hypothesis=False)

                    plot_independent_variables_vs_other(data=self.data,
                                                        column_name=index,
                                                        variable=self.residuals,
                                                        y_label="residuals",
                                                        axhline=True)

            check_fulfill_ratio(true_fulfill_ratio=true_ratio,
                                min_fulfill_ratio=self.min_fulfill_ratio
                                )

        return check_fulfill_ratio(true_fulfill_ratio=true_ratio,
                                   min_fulfill_ratio=self.min_fulfill_ratio,
                                   print_outcome=False)

    def check_error_term_autocorrelation(self) -> bool:
        """
        Checks correlation between the observations of error term by:
        - Durbin-Watson's statistical test,
        - Breusch-Godfrey's statistical test.
        If:
         - silent_mode = True, method returns:
                                              a) True (which means that the assumption is
                                                 fulfilled) if the percentage of statistical tests
                                                 for which the assumption is fulfilled is higher
                                                 than or equal to set min_fulfill_ratio
                                              b) False (which means that the assumption is not
                                                 fulfilled) if the percentage of statistical tests
                                                 for which the assumption is fulfilled is lower
                                                 than set min_fulfill_ratio
         - silent_mode = False, method returns True/False as above and shows additional statistics,
         descriptions which are helpful in assessing the fulfilment of assumption
        """

        durbin_watson_statistic = durbin_watson(self.residuals)

        bg_test = pd.DataFrame(stats_diag.acorr_breusch_godfrey(self.results)[:2],
                               columns=["value"],
                               index=["Lagrange multiplier statistic", "p-value"])

        true_counts = 0
        lower_threshold_dw_stat = 1.5
        upper_threshold_dw_stat = 2.5
        if lower_threshold_dw_stat < durbin_watson_statistic < upper_threshold_dw_stat:
            true_counts = true_counts + 1

        true_counts = true_counts + test_hypothesis(significance_level=self.alpha,
                                                    p_value=bg_test.iloc[1].value,
                                                    print_outcome=False)

        true_ratio = true_counts / 2

        if not self.silent_mode:

            print(
                Color.BOLD + "Assumption 4. Observations of the error term are uncorrelated with "
                             "each other." + Color.END, "\n")

            print("This assumption affects on: \n",
                  "- prediction \n",
                  "- interpretation.", "\n")

            print("One observation of the error term should not predict the next observation. To "
                  "resolve this issue, you might need to add an independent variable to the model "
                  "that captures this information. Analysts commonly use distributed lag models, "
                  "which use both current values of the dependent variable and past values of "
                  "independent variables.\n")

            print(Color.BOLD + "Durbin-Watson " + Color.END + "statistical test: \n",
                  "If the value of the statistics equals 2 => no serial correlation. \n",
                  "If the value of the statistics equals 0 => strong positive correlation. \n",
                  "If the value of the statistics equals 4 => strong negative correlation. \n")
            print(
                "The value of Durbin-Watson statistic is " +
                f"{np.round(durbin_watson(self.residuals), 4)}\n"
            )
            true_counts = 0
            if durbin_watson_statistic < lower_threshold_dw_stat:
                print("Signs of positive autocorrelation =>" +
                      Color.RED + " Assumption not satisfied" + Color.END + "\n")
            elif durbin_watson_statistic > upper_threshold_dw_stat:
                print("Signs of negative autocorrelation =>" +
                      Color.RED + " Assumption not satisfied" + Color.END + "\n")
            else:
                print("Little to no autocorrelation =>" +
                      Color.GREEN + " Assumption satisfied" + Color.END + "\n")
                true_counts = true_counts + 1

            print(Color.BOLD + "Breusch-Godfrey " + Color.END +
                  "Lagrange Multiplier statistical tests: \n")
            print(bg_test, "\n")

            true_counts = true_counts + test_hypothesis(significance_level=self.alpha,
                                                        p_value=bg_test.iloc[1].value,
                                                        null_hypothesis="there doesn't exist "
                                                                        "autocorrelation in the "
                                                                        "error term.")
            true_ratio = true_counts / 2
            check_fulfill_ratio(true_fulfill_ratio=true_ratio,
                                min_fulfill_ratio=self.min_fulfill_ratio)

        return check_fulfill_ratio(true_fulfill_ratio=true_ratio,
                                   min_fulfill_ratio=self.min_fulfill_ratio,
                                   print_outcome=False
                                   )

    def check_error_term_constant_variance(self) -> bool:
        """
        Checks if the error term has constant variance (there is no heteroscedascity) by:
        - Breusch-Pagan's statistical test,
        - Goldfeld-Quandt's statstical test.
        If:
         - silent_mode = True, method returns:
                                              a) True (which means that the assumption is
                                                 fulfilled) if the percentage of statistical tests
                                                 for which the assumption is fulfilled is higher
                                                 than or equal to set min_fulfill_ratio
                                              b) False (which means that the assumption is not
                                                 fulfilled) if the percentage of statistical tests
                                                 for which the assumption is fulfilled is lower
                                                 than set min_fulfill_ratio
         - silent_mode = False, method returns True/False as above and shows additional statistics,
         descriptions which are helpful in assessing the fulfilment of assumption
        """

        bp_test = pd.DataFrame(sms.het_breuschpagan(self.residuals, self.results.model.exog)[:2],
                               columns=["value"],
                               index=["Lagrange multiplier statistic", "p-value"])
        gq_test = pd.DataFrame(sms.het_goldfeldquandt(self.residuals,
                                                      self.results.model.exog)[:-1],
                               columns=["value"],
                               index=["F statistic", "p-value"])
        heteroscedascity_tests = [bp_test, gq_test]

        true_counts = 0
        for test in heteroscedascity_tests:
            true_counts = true_counts + test_hypothesis(significance_level=self.alpha,
                                                        p_value=test.iloc[1].value,
                                                        print_outcome=False)
        true_ratio = true_counts / 2

        if not self.silent_mode:
            print(Color.BOLD + "Assumption 5. The error term has a constant variance." + Color.END,
                  "\n")

            print("This assumption affects on: \n",
                  "- prediction \n",
                  "- interpretation \n")

            print(
                "Heteroscedasticity does not cause bias in the coefficient estimates, it does "
                "make them less precise. Heteroscedasticity also tends to produce p-values that "
                "are smaller than they should be. If you notice this problem in your model, "
                "you can try one of this solutions to fix it: redefine independent variable to "
                "focus on rates/per capita, try using weighted least squares, experiment with "
                "data transformations (f.g. Box-Cox's/Johnson's transformation).\n")

            print(Color.BOLD + "Breusch-Pagan " + Color.END + "Lagrange Multiplier "
                                                              "statistical test: \n")
            print(bp_test, "\n")

            test_hypothesis(significance_level=self.alpha,
                            p_value=bp_test.iloc[1].value,
                            null_hypothesis="error term's variance is constant.")

            print(Color.BOLD + "Goldfeld-Quandt " + Color.END + "test that examines whether the "
                                                                "residual variance is the same in "
                                                                "two subsamples: \n")

            print(gq_test, "\n")

            test_hypothesis(significance_level=self.alpha,
                            p_value=gq_test.iloc[1].value,
                            null_hypothesis="error term's variance is constant.")

            check_fulfill_ratio(true_fulfill_ratio=true_ratio,
                                min_fulfill_ratio=self.min_fulfill_ratio)

            print("HINT: If you see randomly scattered points => there is no heteroscedascity. \n",
                  "If you see fan or cone pattern => probably there exists heteroscedascity. \n")

            plot_standarized_residuals_vs_fitted(fitted_model=self.results)
            plt.show()

        return check_fulfill_ratio(true_fulfill_ratio=true_ratio,
                                   min_fulfill_ratio=self.min_fulfill_ratio,
                                   print_outcome=False
                                   )

    def check_linear_dependency_between_independent_variables(self) -> bool:
        """
        Checks linear dependency of the independent variables by calculating Variance Inflation
        Factor (the square root of the variance inflation factor indicates how much larger the
        standard error increases compared to if that variable had 0 correlation to other predictor
        variables in the model).
        If:
         - silent_mode = True, method returns:
                                              a) True (which means that the assumption is
                                                 fulfilled) if there isn't any independent variable
                                                 that has VIF higher than set vif_threshold
                                              b) False (which means that the assumption is not
                                                 fulfilled) if there is any independent variable
                                                 that has VIF higher than set vif_threshold
         - silent_mode = False, method returns True/False as above and shows additional statistics,
         descriptions which are helpful in assessing the fulfilment of assumption
        """

        vif_df = pd.DataFrame([variance_inflation_factor(self.data.values, variable)
                               for variable in range(self.data.shape[1])],
                              columns=["VIF"],
                              index=self.data.columns
                              ).drop("Intercept").sort_values(by="VIF", ascending=False)

        true_counts = 0
        if vif_df.values.max() < self.vif_threshold:
            true_counts = true_counts + 1

        true_ratio = true_counts

        if not self.silent_mode:
            print(
                Color.BOLD + "Assumption 6. No independent variable is a perfect linear function "
                             "of other explanatory variables (no multicollinearity)." +
                Color.END, "\n")
            print("This assumption affects on: \n",
                  "- interpretation \n")

            print("Multicollinearity makes it hard to interpret your coefficients, and it reduces "
                  "the power of your model to identify independent variables that are "
                  "statistically significant. In case of appearance of this problem you can try "
                  "removing some of the highly correlated variables, linearly combine the "
                  "independent variables, experiment with other analyticla solution (f.g. "
                  "principal components analysis, LASSO or Ridge regression.)\n")

            r2_for_vif_threshold = 1 - 1 / self.vif_threshold
            print(f"For a chosen independent variable VIF > {self.vif_threshold} when the rest of "
                  f"independent variables explain this variable by linear regression model with "
                  f"R^2 > {r2_for_vif_threshold:.2f} => there exists multicollinearity. \n")
            print(
                "Below you can see the values of the " + Color.BOLD + "Variance Inflation Factor "
                + Color.END + "statistic (variables in " + Color.GREEN + "green " + Color.END +
                f"have VIF lower than {self.vif_threshold}, variables in " + Color.RED + "red " +
                Color.END + f"have VIF higher than {self.vif_threshold}): \n")

            if len(self.data.columns) > self.max_iv_show:
                print(f"Due to the fact that the dataset contains more than {self.max_iv_show} "
                      f"independent variables, there will be showed {self.max_iv_show} variables "
                      f"with the highest VIF.\n"
                      "- If you want to see Variance Inflation Factor for more independent "
                      "variables, please change the class' parameter: "
                      "max_independent_variables_to_show to the required number.\n")
                for index, row in vif_df.head(self.max_iv_show).iterrows():
                    if row["VIF"] > self.vif_threshold:
                        print(f"{index}:" + Color.RED + f"{row['VIF']:.2f}" + Color.END)
                    else:
                        print(f"{index}:" + Color.GREEN + f"{row['VIF']:.2f}" + Color.END)
            else:
                for index, row in vif_df.iterrows():
                    if row["VIF"] > self.vif_threshold:
                        print(f"{index}:" + Color.RED + f"{row['VIF']:.2f}" + Color.END)
                    else:
                        print(f"{index}:" + Color.GREEN + f"{row['VIF']:.2f}" + Color.END)

            check_fulfill_ratio(true_fulfill_ratio=true_ratio,
                                min_fulfill_ratio=self.min_fulfill_ratio
                                )

        return check_fulfill_ratio(true_fulfill_ratio=true_ratio,
                                   min_fulfill_ratio=self.min_fulfill_ratio,
                                   print_outcome=False
                                   )

    def check_error_term_normality(self) -> bool:
        """
        Checks if the distribution of the error term is normal by:
        - Shapiro-Wilk's normality test,
        - Jarque-Bera's normality test,
        - Omnibus' normality test,
        - Kolmogorov-Smirnov's normality test,
        - Q-Q plot.
        If:
         - silent_mode = True, method returns:
                                              a) True (which means that the assumption is
                                                 fulfilled) if the percentage of statistical tests
                                                 for which the assumption is fulfilled is higher
                                                 than or equal to set min_fulfill_ratio
                                              b) False (which means that the assumption is not
                                                 fulfilled) if the percentage of statistical tests
                                                 for which the assumption is fulfilled is lower
                                                 than set min_fulfill_ratio
         - silent_mode = False, method returns True/False as above and shows additional statistics,
         descriptions which are helpful in assessing the fulfilment of assumption
        """

        sw = stats.shapiro(self.residuals)
        jb = stats.jarque_bera(self.residuals)
        om = omni_normtest(self.residuals)
        ks = stats.kstest(self.residuals, "norm")
        ad = stats.anderson(self.residuals, dist="norm")
        normality_tests_names = ["Shapiro-Wilk",
                                 "Jarque-Bera",
                                 "Omnibus",
                                 "Kolmogorov-Smirnov"]
        normality_tests = [sw, jb, om, ks, ad]
        tests = zip(normality_tests_names, normality_tests)

        if not self.silent_mode:
            print(Color.BOLD + "Assumption 7. The error term is normally distributed." + Color.END,
                  "\n")
            print("This assumption affects on: \n",
                  "- interpretation \n")

            print("REMARK: For datasets with sufficiently large sample size, the normality of "
                  "errors distribution comes from Central Limit Theorem.\n")

            print("OLS does not require that the error term follows a normal distribution to "
                  "produce unbiased estimates with the minimum variance. However, satisfying this "
                  "assumption allows you to perform statistical hypothesis testing and generate "
                  "reliable confidence intervals and prediction intervals. \n")

            print("Statistical tests for checking normality of the error term distribution: \n")

            true_counts = 0

            for test in tests:
                print(Color.BOLD + f"{test[0]}: " + Color.END +
                      f"test statistic: {test[1][0]:.4f}, p-value: {test[1][1]}")

                true_counts = true_counts + test_hypothesis(
                    self.alpha,
                    p_value=test[1][1],
                    null_hypothesis="the error term is normally distributed"
                )

            true_ratio = true_counts / len(normality_tests)

            check_fulfill_ratio(true_fulfill_ratio=true_ratio,
                                min_fulfill_ratio=self.min_fulfill_ratio)

            print("Q-Q (quantile-quantile) plot: \n")
            print(
                "HINT: If error term's distribution is similar to normal distribution, the points "
                "in the Q–Q plot will approximately lie on the line y = x")
            sm.qqplot(self.residuals, line='s')
            plt.show()
        else:
            true_counts = 0

            for test in tests:
                true_counts = true_counts + test_hypothesis(
                    self.alpha,
                    p_value=test[1][1],
                    print_outcome=False
                )

            true_ratio = true_counts / len(normality_tests)

        return check_fulfill_ratio(true_fulfill_ratio=true_ratio,
                                   min_fulfill_ratio=self.min_fulfill_ratio,
                                   print_outcome=False
                                   )

    def check_all_assumptions(self) -> bool:
        """Checks all OLS regression assumptions.
        If:
         - silent_mode = True, method returns:
                                              a) True (which means that the assumption is
                                                 fulfilled) if all 1-7 assumptions are fulfilled
                                              b) False (which means that the assumption is not
                                                 fulfilled) if any of 1-7 assumptions is not
                                                 fulfilled
         - silent_mode = False, method returns True/False as above and shows additional statistics,
         descriptions and plots which are helpful in assessing the fulfilment of every single
         assumption
        """

        if not self.silent_mode:
            print(Color.BOLD + "Checking the assumptions of OLS regression" + Color.END, "\n")
            print("In each of the following assumptions that will be checked, the following "
                  "conventions have been adopted: \n" +
                  Color.GREEN + "- the green color " + Color.END + "means that the assumption is "
                                                                   "fulfilled,\n" +
                  Color.RED + "- the red color " + Color.END + "means that the assumption is not "
                                                               "fulfilled.\n")

            self.summarize_model()

        return all([self.check_linearity(),
                    self.check_error_term_zero_mean(),
                    self.check_independent_variables_vs_error_correlation(),
                    self.check_error_term_autocorrelation(),
                    self.check_error_term_constant_variance(),
                    self.check_linear_dependency_between_independent_variables(),
                    self.check_error_term_normality()
                    ])
