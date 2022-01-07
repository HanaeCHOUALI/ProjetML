import numpy as np
from sklearn.decomposition import KernelPCA


class PrincipalComponentAnalysis:
    def __init__(self, n_components, x_train, x_test):
        self.n_components = n_components
        self.x_train = x_train
        self.x_test = x_test

        self.pca = None

    def fit(self, scaler):
        self.x_train = scaler.fit_transform(self.x_train)
        self.pca = KernelPCA(n_components=self.n_components, kernel='poly')

        self.pca.fit(self.x_train)
        self.x_train = self.pca.transform(self.x_train)
        self.x_test = self.pca.transform(self.x_test)

    def test(self):
        print('Explained variance ratio ', self.pca.explained_variance_ratio_[:300])

        var = np.cumsum(self.pca.explained_variance_ratio_[:300])
        print('Cumulative explained variance ratio ', var)
        pass
