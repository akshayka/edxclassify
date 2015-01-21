'''

INPUT: string --> classifier
OUTPUT: subclass instance of Classifier

'''

from classifiers.linear_svc import LinSVC
from classifiers.log_regression import LogRegression
from classifiers.naive_bayes import NaiveBayes
from classifiers.custom_token_patterns import CUSTOM_TOKEN_PATTERNS


supported_classifiers = 'naive_bayes\nlogistic\nlin_svc'

def make_classifier(clf, reduce_features=False, k_best=0, token_pattern_idx=0,
                    text_only=False, no_text=False,
                    tfidf=False, penalty=1.0):

    if token_pattern_idx >= len(CUSTOM_TOKEN_PATTERNS):
        raise NotImplementedError('Token pattern %d not implemented'
                                  % token_pattern_idx)

    clf = clf.lower()
    # This class is trivial, currently; it will only be useful if
    # a custom, non-sklearn classifier is rolled
    return SklearnCLF(
                  clf, 
                  token_pattern=CUSTOM_TOKEN_PATTERNS[token_pattern_idx],
                  text_only=text_only,
                  no_text=no_text,
                  tfidf=tfidf,
                  C=penalty,
                  reduce_features=reduce_features,
                  k_best_features=k_best)

        raise NotImplementedError('Classifier %s not supported; choose from:\n'
                                  '%s' % (clf, supported_classifiers))
