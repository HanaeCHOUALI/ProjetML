from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


class RandomForestModel:
    def __init__(self, x_train, x_test, y_train, y_test, parameters):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.parameters = parameters
        self.clf = None

    def train(self, scaler, cv=3):
        pipe = Pipeline([('scaler', scaler), ('rf', RandomForestClassifier())])

        self.clf = GridSearchCV(pipe, self.parameters, cv=cv)
        self.clf.fit(self.x_train, self.y_train)

    def test(self):
        print('Returned hyperparameter: {}'.format(self.clf.best_params_))
        print('Best accuracy in train is: {}'.format(self.clf.best_score_))
        print('Classification accuracy on test is: {}'.format(self.clf.score(self.x_test, self.y_test)))
