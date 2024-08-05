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

| Hyperparameter     | Value  |
|--------------------|--------|
| num rounds         | 464    |
| eta                | 0.0082 |
| subsample          | 0.982  |
| max depth          | 11     |
| min child weight   | 3.30   |
| colsample bytree   | 0.975  |
| colsample bylevel  | 0.900  |
| lambda             | 0.06068|
| alpha              | 0.00235|
| gamma              | 0      |

For on overview of what these hyperparameters do see:
https://xgboost.readthedocs.io/en/stable/parameter.html.

**NOTE:** the names of these hyperparameters might not match the names of the hyperparameters of the XGBoost API you are using, e.g., num rounds refers to n estimators in the python (scikitlearn) XGBoost API whereas it refers to nrounds in the R (mlr3) XGBoost API. Be careful and check the documentation!

These 20 OpenML tasks should be used for evaluation of your solution:
16, 22, 31, 2074, 2079, 3493, 3907, 3913, 9950, 9952, 9971, 10106, 14954, 14970, 146212, 146825,167119, 167125, 168332, 168336

**NOTE:** These tasks naturally come with a 10-fold CV resampling. To speed things up, you can and should train only on the train set of the first fold and evaluate on the test set of the first fold!

The meta data can be found here:
https://syncandshare.lrz.de/getlink/fiV9MfvupyNzWpT99M5RhFh2/

  - xgboost meta data.csv contains 3.386.866 evaluations of configurations across 99 tasks.
  - features.csv contains simple pre-computed meta features for the evaluated datasets.
  
To this end, you could consider the following:
  •  Try out different performance predictors (EPMs).
  • Evaluate which meta-features are useful.
  • Evaluate which hyperparameters have the most influence on the performance.
  • Try to find the simplest meta-learning approach that beats the default configuration.
  • Check if you can find an improved static default configuration.
  
**Important:** Do not overfit on the 20 test tasks. E.g., to evaluate the performance of different EPMs use the precomputed data.












