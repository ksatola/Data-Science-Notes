# Check OLS Assumptions

Ordinary Least Squares is the most common estimation method for linear models. 
The purpose of this package is to help in checking the assumptions of OLS Linear 
Regression. If the assumptions hold true, the OLS procedure creates the best possible
estimates. In statistics, estimators that produce unbiased estimates that have the 
smallest variance are referred to as being “efficient.” Efficiency is a statistical 
concept that compares the quality of the estimates calculated by different procedures 
while holding the sample size constant. OLS is the most efficient linear regression 
estimator when the assumptions are satisfied.

### The assumptions of OLS Linear Regression:

1. Regression model is linear in the coefficients and the error term.
2. The error term has mean of 0.
3. All independent variables are uncorrelated with the error term.
4. Observations of the error term are uncorrelated with each other.
5. The error term has a constant variance.
6. No independent variable is a perfect linear function of other explanatory variables.
7. The error term is normally distributed.

With the provided package you can check all mentioned assumptions.
Firstly you need to instantiate the class:
```
colsa = CheckOLSAssumptions(data, 
                            target, 
                            significance_level, 
                            vif_threshold, 
                            max_independent_variables_to_show,
                            min_fulfill_ratio,
                            silent_mode)
```

Class can be used both for:
 - checking the assumptions by analysing several statistics and plots (`silent_mode=False`)
 - checking the assumptions by looking only on the boolean variable that is returned 
   (`silent_mode=True`)

Then, if you want to check all the assumptions at once:
```
colsa.check_all_assumptions()
```

If you want to check the assumptions separately:
1. Regression model is linear in the coefficients and the error term.
    ```
    colsa.check_linearity()
    ```
2. The error term has mean of 0.
    ```
    colsa.check_error_term_zero_mean()
    ```
3. All independent variables are uncorrelated with the error term.
    ```
    colsa.check_independent_variables_vs_error_correlation()
    ```
4. Observations of the error term are uncorrelated with each other.
    ```
    colsa.check_error_term_autocorrelation()
    ```
5. The error term has a constant variance.
    ```
    colsa.check_error_term_constant_variance()
    ```
6. No independent variable is a perfect linear function of other explanatory variables.
    ```
    colsa.check_linear_dependency_between_independent_variables()
    ```
7. The error term is normally distributed.
    ```
    colsa.check_error_term_normality()
    ```
   