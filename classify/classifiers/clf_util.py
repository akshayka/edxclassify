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
        feature_names.extend([name + "__" + f for f in
                                        dv.get_feature_names()])
    return np.asarray(feature_names)


def sklearn_cv(clf, examples, labels):
    X, y = np.array(examples), np.array(labels)
    skf = StratifiedKFold(labels, n_folds=10)
    precision_train = []
    recall_train = []
    f1_train = []
    precision_test = []
    recall_test = []
    f1_test = []
    relevant_features = {'knowledgeable': [], 'neutral': [], 'confused': []}
    for train_indices, test_indices in skf:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        # Predict on the test set
        print 'fitting ...'
        clf.fit(X_train, y_train)
        print 'predicting ...'
        y_pred = clf.predict(X_test)

        # Extract information about the most relevant features
        feature_names = extract_feature_names(clf.steps[0][-1])
        classifier = clf.steps[-1][-1]
        if feature_names is not None and hasattr(classifier, 'coef_'):
            for i, label in enumerate(['knowledgeable', 'neutral', 'confused']):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                relevant_features[label].append([feature_names[top10]])

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

