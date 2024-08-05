# Python Code Skeleton for Meta-Learning XGBoost Use-Case

### Setup environment

If using _Linux_ with Python and _anaconda/miniconda_ already installed, the following script should setup the required environment:
```commandline
bash setup_environment.sh
```

For a more general installation of dependencies:
```commandline
pip install -r requirements.txt
```

### Testing setup
```commandline
python metalearning_xgboost/baseline.py
```


### Meta-training tips
* Download the data from [here](https://syncandshare.lrz.de/getlink/fiV9MfvupyNzWpT99M5RhFh2/)
* Load the `meta_features` and `meta_data` with the help of `load_data_from_path()` in utils
* The list of hyperparameters that need to be considered from the `meta_data` can be retrieved using `get_hyperparameter_list()` in utils
* The list of metafeature variables that need to be considered from `meta_features` can be retrieved using `get_metafeature_list()` in utils
* Utilize the `XGBoostTest` class in utils in order to load the task data once and do multiple evaluations with different configurations


### Evaluating on a test task
```python
from utils import XGBoostTest, test_ids, default_config


seed=123
task_id = 16  # any Task ID from test_ids
# create the XGBoost interface
objective = XGBoostTest(task_id, seed)
# extract meta features
meta_features = objective.meta_features
# pass the meta feature to your model for a configuration
config = your_model(objective.meta_features)
# evaluate the config
auc, traintime = objective.evaluate(config)
# evaluate the default config for the baseline
base_auc, base_traintime = objective.evaluate(default_config)
```

**Important**: You don't have to use the provided code, feel free to start from scratch with your own code. This is merely meant as a (potential) starting point
