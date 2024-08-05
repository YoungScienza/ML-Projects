import os
import openml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

test_ids = [
    16, 22, 31, 2074, 2079, 3493, 3907, 3913, 9950, 9952, 9971, 10106, 14954, 14970, 146212,
    146825, 167119, 167125, 168332, 168336
]

meta_feature_names = [
    "data_id", "name", "status", "MajorityClassSize", "MaxNominalAttDistinctValues",
    "MinorityClassSize", "NumberOfClasses", "NumberOfFeatures", "NumberOfInstances",
    "NumberOfInstancesWithMissingValues", "NumberOfMissingValues", "NumberOfNumericFeatures",
    "NumberOfSymbolicFeatures"
]

default_config = {
    "n_estimators": 464,  # num_rounds
    "eta": 0.0082,
    "subsample": 0.982,
    "max_depth": 11,
    "min_child_weight": 3.30,
    "colsample_bytree": 0.975,
    "colsample_bylevel": 0.9,
    "lambda": 0.06068,
    "alpha": 0.00235,
    "gamma": 0
}

def get_hyperparameter_list() -> List[str]:
    return list(default_config.keys())

def get_metafeature_list() -> List[str]:
    return meta_feature_names

def load_data_from_path(path_to_files: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the meta files from disk and returns them as data frames"""
    files_to_load = dict(
        features="../features.csv",
        meta_features="../xgboost_meta_data.csv"
    )
    meta_features = pd.read_csv(os.path.join(path_to_files, files_to_load["features"]))
    meta_data = pd.read_csv(os.path.join(path_to_files, files_to_load["meta_features"]))
    return meta_features, meta_data

def _get_preprocessor(categoricals, continuous):
    """Preprocessing"""
    preprocessor = make_pipeline(
        ColumnTransformer([
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore")
                ),
                categoricals.tolist(),
            ),
            (
                "cont",
                make_pipeline(
                    SimpleImputer(strategy="median")
                ),
                continuous.tolist(),
            )
        ])
    )
    return preprocessor

def get_task_metafeatures(task_id: int, meta_feature_names: List[str]) -> Dict:
    """Get meta features from an OpenML task based on its task id"""
    task = openml.tasks.get_task(task_id)
    features = openml.datasets.list_datasets(data_id=[task.dataset_id])[task.dataset_id]
    features["data_id"] = features["did"]

    for feature in set(features.keys()) - set(meta_feature_names):
        features.pop(feature)

    return features

def _convert_labels(labels):
    """Converts boolean labels (if exists) to strings"""
    label_types = list(map(lambda x: isinstance(x, bool), labels))
    if np.all(label_types):
        _labels = list(map(lambda x: str(x), labels))
        if isinstance(labels, pd.Series):
            labels = pd.Series(_labels, index=labels.index)
        elif isinstance(labels, np.array):
            labels = np.array(labels)
    return labels

def load_test_data(task_id: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetches data from OpenML, converts data types, to yield train-test numpy arrays"""
    task = openml.tasks.get_task(task_id, download_data=False)
    nclasses = len(task.class_labels)
    dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
    X, y, categorical_ind, feature_names = dataset.get_data(target=task.target_name, dataset_format="dataframe")

    categorical_ind = np.array(categorical_ind)
    (cat_idx,) = np.where(categorical_ind)
    (cont_idx,) = np.where(~categorical_ind)

    # splitting dataset into train and test (10% test)
    # train-test split is fixed for a task and its associated dataset (from OpenML)
    train_idx, test_idx = task.get_train_test_split_indices()  # we only use the first of the 10 CV folds

    train_x = X.iloc[train_idx]
    train_y = y.iloc[train_idx]
    test_x = X.iloc[test_idx]
    test_y = y.iloc[test_idx]

    preprocessor = _get_preprocessor(cat_idx, cont_idx)

    # preprocessor fit only on the training set
    train_x = preprocessor.fit_transform(train_x)
    test_x = preprocessor.transform(test_x)

    # converting bool labels
    train_y = _convert_labels(train_y)
    test_y = _convert_labels(test_y)

    # encoding labels
    le = LabelEncoder()
    le.fit(np.unique(train_y))
    train_y = le.transform(train_y)
    test_y = le.transform(test_y)

    return train_x, train_y, test_x, test_y, nclasses
