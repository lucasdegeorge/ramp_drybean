import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit

import rampwf as rw

problem_title = "Dry Bean Species Classification"

# Mapping int to classes
n_classes = 7
int_to_cat = {
    0: 'SEKER', 
    1: 'BARBUNYA',
    2: 'BOMBAY',
    3: 'CALI',
    4: 'HOROZ',
    5: 'SIRA', 
    6: 'DERMASON'
}

# Mapping categories to int
cat_to_int = {v: k for k, v in int_to_cat.items()}

_event_label_int = list(int_to_cat)

Predictions = rw.prediction_types.make_multiclass(label_names=_event_label_int)
workflow = rw.workflows.Classifier()


score_types = [
    rw.score_types.Accuracy(precision=3),
    rw.score_types.BalancedAccuracy(precision=3)
]

# Create global variable to use in LOGO CV strategy
groups = None


def _get_data(path=".", split="train"):
    """ Load data from csv file.
        Returns X (input) and y (output) arrays
            X: array of shape (n_samples, n_features)
            y: array of shape (n_samples, n_classes)
    """
    data = pd.read_csv(os.path.join(path, "data", split + ".csv"))
    # features
    X = data.drop("Class", axis=1).to_numpy()
    # label
    y = data['Class'].apply(lambda x: cat_to_int[x]).to_numpy()

    return X, y

def get_train_data(path="."):
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")

def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)