import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics


def sklearn_cv(clf, examples, labels):
    X, y = np.array(examples), np.array(labels)
    skf = StratifiedKFold(labels, n_folds=10)
    precision_train = []
    recall_train = []
    f1_train = []
    precision_test = []
    recall_test = []
    f1_test = []
    for train_indices, test_indices in skf:
        X_train, X_test  = X[train_indices], X[test_indices]
        y_train, y_test  = y[train_indices], y[test_indices]
        # Predict on the test set
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        precision_test.append(metrics.precision_score(y_test, y_pred,
            average=None))
        recall_test.append(metrics.recall_score(y_test, y_pred, average=None))
        f1_test.append(metrics.f1_score(y_test, y_pred, average=None))

        # Predict on the training set
        y_pred = clf.predict(X_train)
        precision_train.append(metrics.precision_score(y_train, y_pred,
            average=None))
        recall_train.append(metrics.recall_score(y_train, y_pred, average=None))
        f1_train.append(metrics.f1_score(y_train, y_pred, average=None))
    return [precision_train, recall_train, f1_train], [precision_test, recall_test, f1_test]
