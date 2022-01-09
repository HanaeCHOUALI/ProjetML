import numpy as np


class PrincipalComponentAnalysis:
    def __init__(self, model, x_train, x_test, kernel=False):
        self.x_train = x_train
        self.x_test = x_test

        self.model = model
        self.kernel = kernel

    def fit(self, scaler):
        self.x_train = scaler.fit_transform(self.x_train)

        self.model.fit(self.x_train)
        self.x_train = self.model.transform(self.x_train)
        self.x_test = self.model.transform(self.x_test)

    def test(self):
        if not self.kernel:
            print('Explained variance ratio ', self.model.explained_variance_ratio_)
            var = np.cumsum(self.model.explained_variance_ratio_)
            print('Cumulative explained variance ratio ', var)
