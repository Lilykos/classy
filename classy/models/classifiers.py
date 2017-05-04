from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from ..utils import log

classifier_names = {'multinomialnb': 'Multinomial NB', 'knn': 'KNN', 'adaboost': 'AdaBoost', 'linearsvc': 'Linear SVC',
                    'svc': 'SVC', 'nb': 'Naive Bayes', 'randomforest': 'Random Forest', 'logisticregression': 'Logistic Regression'}


@log('Classifier: Multinomial Naive Bayes')
def multinomialnb(X, y):
    return MultinomialNB().fit(X, y)


@log('Classifier: Gaussian Naive Bayes')
def nb(X, y):
    return GaussianNB().fit(X, y)


@log('Classifier: K Nearest Neighbors')
def knn(X, y):
    return KNeighborsClassifier().fit(X, y)


@log('Classifier: Random Forest')
def random_forest(X, y):
    return RandomForestClassifier().fit(X, y)


@log('Classifier: Logistic Regression')
def logistic_regression(X, y):
    return LogisticRegression().fit(X, y)


# @log('Classifier: Linear SVC')
# def linearsvc(X, y):
#     return LinearSVC().fit(X, y)


@log('Classifier: SVC')
def svc(X, y):
    return SVC(probability=True).fit(X, y)


@log('Classifier: AdaBoost')
def adaboost(X, y):
    return AdaBoostClassifier().fit(X, y)


classifiers = {
    'multinomialnb': multinomialnb,
    'nb': nb,
    'knn': knn,
    'randomforest': random_forest,
    'logisticregression': logistic_regression,
    # 'linearsvc': linearsvc,
    'svc': svc,
    'adaboost': adaboost
}
