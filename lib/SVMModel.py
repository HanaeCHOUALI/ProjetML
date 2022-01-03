from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC


class SVMModel:
    def __init__(self, x_train, x_test, y_train, y_test, parameters):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.parameters = parameters
        self.clf = None

    def train(self, scaler, max_iter=3000, cv=3):
        pipe = Pipeline([('scaler', scaler), ('svc', LinearSVC(max_iter=max_iter))])

        self.clf = GridSearchCV(pipe, self.parameters, cv=cv)
        self.clf.fit(self.x_train, self.y_train)

    def test(self):
        print('\nModel: Support Vector Machine')
        print('Returned hyperparameter: {}'.format(self.clf.best_params_))
        print('Best accuracy in train is: {}'.format(self.clf.best_score_))
        print('Classification accuracy on test is: {}'.format(self.clf.score(self.x_test, self.y_test)))
