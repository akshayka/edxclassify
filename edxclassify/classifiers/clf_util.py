import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn.externals import joblib
import skll


def load_clf(pkl_file):
    """Load a joblib-dumped data_cleaner and trained classifier"""
    data_cleaner, clf = joblib.load(pkl_file)
    return data_cleaner, clf


def extract_feature_names(feature_union):
    pipelines = feature_union.transformer_list
    feature_names = []
    for name, pipeline in pipelines:
        dv = pipeline.steps[-1][-1]
        if not hasattr(dv, 'get_feature_names'):
            raise AttributeError("Dictionary %s does not provide "
                                 "get_feature_names." % str(name))
        feature_names.extend([name + ' ' + f for f in
                                        dv.get_feature_names()])
    return np.asarray(feature_names)


def sklearn_cv(clf, X, y):
    """Evaluate training and test set error using stratified K-fold
    cross validation.

    parameters:
    ----------
    clf        - a scikit-learn pipelined estimator.
    X          - a list of feature vectors
    y          - a list of labels, with y[i] the label for X[i]

    returns:
    --------
    train_error_metrics: A list that itself contains four lists, p, r, f, K,
                        each with length 10 (corresponding to the number of
                        folds):
                            Element i in p is a list whose jth element is the
                            precision for class j in the ith fold;
                            Element i in r is a list whose jth element is the
                            recall for class j in the ith fold;
                            Element i in f is a list whose jth element is the
                            f1 for class j in the ith fold; and
                            Element i in K is the Kappa Coefficient for the
                            ith fold.
    test_error_metrics: Like train_error_metrics, but for the test_set_error.
    """

    X, y = np.array(X), np.array(y)
    skf = StratifiedKFold(y, n_folds=10)
    precision_train = []
    recall_train = []
    f1_train = []
    kappa_train = [] 
    precision_test = []
    recall_test = []
    f1_test = []
    kappa_test = [] 
    
    for train_indices, test_indices in skf:
        print 'cross_validating ...'
        # Partition the dataset, as per the fold partitioning.
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Train the classifier and
        # extract the most highly weighted features.
        clf.fit(X_train, y_train)

        # Predict labels for the train set.
        y_pred = clf.predict(X_train)
        precision_train.append(metrics.precision_score(y_train, y_pred,
            average=None))
        recall_train.append(metrics.recall_score(y_train, y_pred, average=None))
        f1_train.append(metrics.f1_score(y_train, y_pred, average=None))
        kappa_train.append(skll.metrics.kappa(y_train, y_pred))

        # Predict labels for the test set.
        y_pred = clf.predict(X_test)
        precision_test.append(metrics.precision_score(y_test, y_pred,
            average=None))
        recall_test.append(metrics.recall_score(y_test, y_pred, average=None))
        f1_test.append(metrics.f1_score(y_test, y_pred, average=None))
        kappa_test.append(skll.metrics.kappa(y_test, y_pred))

    return [precision_train, recall_train, f1_train, kappa_train],\
            [precision_test, recall_test, f1_test, kappa_test]
