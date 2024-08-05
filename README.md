# AML-Project
This is my project about meta learning for XGBoost for the exam of Automated ML.

## Meta-Learning for XGBoost hyperparameters
Extreme Gradient Boosting (XGBoost) is an efficient implementation of gradient boosted trees.
It is highly configurable by a number of hyperparameters that influence the performance. In
this project we provide you with a large dataset of XGBoost configurations evaluated on 99
classification data sets.
Your task is to develop a meta-learning approach using that data to learn which configurations to
use on a set of 20 new classification problems.
Your meta-learner has to outperform the following default configuration:

| Hyperparameter | Value       |
|----------------|-------------|
|num rounds      | 464         |
|eta             | 0.0082 |
|subsample |0.982|
|max depth |11|
|min child weight |3.30|
|colsample bytree | 0.975|
|colsample bylevel | 0.900|
|lambda |0.06068|
|alpha |0.00235|
|gamma |0|
For on overview of what these hyperparameters do see:
https://xgboost.readthedocs.io/en/stable/parameter.html.



