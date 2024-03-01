from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class Classifier(BaseEstimator):
    def __init__(self):
        self.transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        self.model = KNeighborsClassifier()
        self.n_classes = 7

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        pred = np.random.randint(0, self.n_classes, X.shape[0])
        return pred 

    def predict_proba(self, X):
        proba = np.random.rand(X.shape[0], 7)
        proba = np.exp(proba)/sum(np.exp(proba))
        return proba