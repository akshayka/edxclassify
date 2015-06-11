"""
TODO: Describe how to use this, default clfrs, schemas,
      etc.
"""

import edxclassify.classifiers.clf_util
import package_resources

DEFAULT_CLF = [
    'confusion',
    'confusion_statistics',
    'confusion_economics',
    'confusion_medicine',
    'sentiment',
    'sentiment_statistics',
    'sentiment_economics',
    'sentiment_medicine',
    'opinion',
    'opinion_statistics',
    'opinion_economics',
    'opinion_medicine',
    'answer',
    'answer_statistics',
    'answer_economics',
    'answer_medicine',
    'question',
    'question_statistics',
    'question_economics',
    'question_medicine',
    'urgency',
    'urgency_statistics',
    'urgency_economics',
    'urgency_medicine',
]

class LiveCLF:
    def __init__(self, clf_key):
        # clf_key is either a path or a keyword mapping to default location
        if clf_key in DEFAULT_CLF:
            clf_key = package_resources.resource_string('classify', 'saved_clf')
            clf_key = clf_key + '/' + clf_key
        self.name = clf_key
        dc, clf = clf_util.load_clf(clf_key)
        self.dc = dc
        self.clf = clf

    def predict(self, examples):
        """Generate predictions for examples using underlying classifier

        parameters:
        ----------
        examples - a dictionary laid out as per the feature specification
                   of the underlying classifier
        
        returns:
        --------
        an array of shape [n_examples], with one prediction for each example
        """
        data = [e.values() for e in examples]
        X = self.dc.process_records_without_labels(data)
        return self.clf.test(X, None)
