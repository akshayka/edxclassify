import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics


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


def sklearn_cv(clf, X, y, str_labels):
    """Evaluate training and test set error using stratified K-fold
    cross validation.

    Parameters:
    ----------
    clf        - a scikit-learn pipelined estimator.
    X          - a list of feature vectors
    y          - a list of labels, with y[i] the label for X[i]
    str_labels - a list of string identifiers, one for each label

    Returns
    -------
    train_error_metrics: A list that itself contains 10 lists, one for each
                    fold, describing training error.
                    Element i is a list [p, r, f], where
                        p is a tuple whose jth element is the precision for
                        label j,
                        r is a tuple whose jth element is the recall for label j,
                        and f is a tuple whose jth element is the f1 for label j.
    test_error_metrics: Like train_error_metrics, but for the test_set_error.
    relevant_features: A dictionary describing informative features.
                    In particular, if clf is a multiclass classifier,
                    relevant_features maps each string label to a list
                    containing the 600 features weighted most highly by the
                    classifier's decision function(s).
                    If clf is a binary class, then a dictionary mapping
                    the word 'informative' to the list of 600 features weighted
                    most highly by the classifier's decision function.
    """

    X, y = np.array(X), np.array(y)
    skf = StratifiedKFold(y, n_folds=10)
    precision_train = []
    recall_train = []
    f1_train = []
    precision_test = []
    recall_test = []
    f1_test = []
    relevant_features = {}
    num_top = 600
    
    if len(str_labels) > 2:
        for label in str_labels:
            relevant_features[label] = []
    else:
        relevant_features['informative'] = []

    for train_indices, test_indices in skf:
        print 'cross_validating ...'
        # Partition the dataset, as per the fold partitioning.
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Train the classifier and
        # extract the most highly weighted features.
        clf.fit(X_train, y_train)
        feature_names = extract_feature_names(clf.steps[0][-1])

        # A bit of a hack -- we index into the pipeline in order to
        # retrieve the actual estimator.
        classifier = clf.steps[-1][-1]
        if feature_names is not None and hasattr(classifier, 'coef_'):
            if len(str_labels) == 2:
                relevant_features['informative'] =\
                    np.argsort(classifier.coef_[0])[-num_top:]
            else:
                for i, label in enumerate(str_labels):
                    top = np.argsort(classifier.coef_[i])[-num_top:]
                    relevant_features[label].append(feature_names[top])

        # Predict labels for the training set.
        y_pred = clf.predict(X_test)
        precision_test.append(metrics.precision_score(y_test, y_pred,
            average=None))
        recall_test.append(metrics.recall_score(y_test, y_pred, average=None))
        f1_test.append(metrics.f1_score(y_test, y_pred, average=None))

        # Predict labels for the test set.
        y_pred = clf.predict(X_train)
        precision_train.append(metrics.precision_score(y_train, y_pred,
            average=None))
        recall_train.append(metrics.recall_score(y_train, y_pred, average=None))
        f1_train.append(metrics.f1_score(y_train, y_pred, average=None))

    return [precision_train, recall_train, f1_train],\
           [precision_test, recall_test, f1_test], relevant_features
