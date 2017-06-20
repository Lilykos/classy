from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.base import TransformerMixin
from ..utils import log

classifier_names = {'knn': 'KNN', 'adaboost': 'AdaBoost', 'linearsvc': 'Linear SVC', 'svc': 'SVC', 'nb': 'Naive Bayes',
                    'randomforest': 'Random Forest', 'logisticregression': 'Logistic Regression'}


class ClassifierSelector(TransformerMixin):
    @log('Classifier: Naive Bayes')
    def _nb(self, X, y):
        return GaussianNB().fit(X, y)

    @log('Classifier: K Nearest Neighbors')
    def _knn(self, X, y):
        return KNeighborsClassifier().fit(X, y)

    @log('Classifier: Random Forest')
    def _random_forest(self, X, y):
        return RandomForestClassifier().fit(X, y)

    @log('Classifier: Logistic Regression')
    def _logistic_regression(self, X, y):
        return LogisticRegression().fit(X, y)

    @log('Classifier: SVC')
    def _svc(self, X, y):
        return SVC(probability=True).fit(X, y)

    @log('Classifier: AdaBoost')
    def _adaboost(self, X, y):
        return AdaBoostClassifier().fit(X, y)

    @log('Classifier: Linear SVC')
    def _linearsvc(self, X, y):
        return LinearSVC().fit(X, y)

    def __init__(self, attrs, algorithm, X_test):
        self.classifiers = {
            'nb': self._nb,
            'knn': self._knn,
            'randomforest': self._random_forest,
            'logisticregression': self._logistic_regression,
            'svc': self._svc,
            'adaboost': self._adaboost,
            'linearsvc': self._linearsvc
        }
        self.binary = True if attrs['labelsType'] == 'binary' else False
        self.algorithm = algorithm
        self.X_test = X_test

    def fit(self, y=None):
        return self

    def transform(self, X, y=None):
        model = self.classifiers[self.algorithm](X, y)
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None

        return model, y_pred, y_prob
