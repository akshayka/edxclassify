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


def sklearn_cv(clf, X, y, labels):
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
    
    if len(labels) > 2:
        for label in labels:
            relevant_features[label] = []
    else:
        relevant_features['informative'] = []

    for train_indices, test_indices in skf:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        # Predict on the test set
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Extract information about the most relevant features
        feature_names = extract_feature_names(clf.steps[0][-1])
        classifier = clf.steps[-1][-1]
        if feature_names is not None and hasattr(classifier, 'coef_'):
            if len(labels) == 2:
                relevant_features['informative'] =\
                    np.argsort(classifier.coef_[0])[-num_top:]
            else:
                for i, label in enumerate(labels):
                    top = np.argsort(classifier.coef_[i])[-num_top:]
                    relevant_features[label].append(feature_names[top])

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
    return [precision_train, recall_train, f1_train], [precision_test, recall_test, f1_test], relevant_features

def to_int(value):
    if value == '':
        return 0
    return int(value)

def to_float(value):
    if value == '':
        return 0
    return 1 if float(value) > 0.94 else 0

def is_anonymous(value):
    return 1 if value.lower() == 'true' else 0

def is_comment_thread(value):
    return 1 if value.lower() == 'commentthread' else 0

# TODO: We might want to discretize the grades and number of attempts

