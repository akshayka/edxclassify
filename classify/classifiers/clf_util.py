import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics


def sklearn_cv(clf, examples, labels):
    X, y = np.array(examples), np.array(labels)
    skf = StratifiedKFold(labels, n_folds=10)
    precision = []
    recall = []
    f1 = []
    for train_indices, test_indices in skf:
        X_train, X_test  = X[train_indices], X[test_indices]
        y_train, y_test  = y[train_indices], y[test_indices]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        precision.append(metrics.precision_score(y_test, y_pred,
            average=None))
        recall.append(metrics.recall_score(y_test, y_pred, average=None))
        f1.append(metrics.f1_score(y_test, y_pred, average=None))
    return [precision, recall, f1]
