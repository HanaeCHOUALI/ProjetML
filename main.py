import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lib.LogisticRegressionModel import LogisticRegressionModel
from lib.RandomForestModel import RandomForestModel
from lib.SVMModel import SVMModel


data = pd.read_csv('data/data.csv', sep=",", index_col=0)
labels = pd.read_csv('data/labels.csv', sep=",", index_col=0)

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Unnamed: 0'], axis=1),
    labels['Class'],
    test_size=0.2,
    random_state=42
    )

# Build Random Forest model
rf_parameters = {'rf__max_depth': np.arange(1, 101, 5)}
rf_model = RandomForestModel(X_train, X_test, y_train, y_test, rf_parameters)
rf_model.train(scaler=StandardScaler(), cv=3)
rf_model.test()

# Build Logistic Regression model
lr_parameters = {'logreg__C': np.logspace(-2, 2, 5, base=2)}
lr_model = LogisticRegressionModel(X_train, X_test, y_train, y_test, lr_parameters)
lr_model.train(StandardScaler(), max_iter=3000, cv=3)
lr_model.test()

# Build SVM model
svm_parameters = {'svc__C': np.logspace(-2, 2, 5, base=2)}
svm_model = SVMModel(X_train, X_test, y_train, y_test, svm_parameters)
svm_model.train(StandardScaler(), max_iter=3000, cv=3)
svm_model.test()

