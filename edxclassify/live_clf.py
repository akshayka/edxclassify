"""
TODO: Describe how to use this, default clfrs, schemas,
      etc.

{
    'text' => ...
    'bleh' => ...
}

arr len 20

arr[feature_cols['text']] = example['text']
"""

import edxclassify.classifiers.clf_util as clf_util
from edxclassify.feature_spec import FEATURE_COLUMNS
import pkg_resources

DEFAULT_CLF = [
    'confusion',
    'confusion_stats',
    'confusion_econ',
    'confusion_medicine',
    'confusion_technical',
    'confusion_nontechnical',
    'confusion_medicine_txt',
    'sentiment',
    'sentiment_stats',
    'sentiment_econ',
    'sentiment_medicine',
    'sentiment_technical',
    'sentiment_nontechnical',
    'opinion',
    'opinion_stats',
    'opinion_econ',
    'opinion_medicine',
    'opinion_technical',
    'opinion_nontechnical',
    'answer',
    'answer_stats',
    'answer_econ',
    'answer_medicine',
    'answer_technical',
    'answer_nontechnical',
    'question',
    'question_stats',
    'question_econ',
    'question_medicine',
    'question_technical',
    'question_nontechnical',
    'urgency',
    'urgency_stats',
    'urgency_econ',
    'urgency_medicine',
    'urgency_technical',
    'urgency_nontechnical',
]

class LiveCLF:
    def __init__(self, clf_key):
        # clf_key is either a path or a keyword mapping to default location
        self.name = clf_key
        if clf_key in DEFAULT_CLF:
            '''clf_key = pkg_resources.resource_string(
                'edxclassify', 'saved_clf' + '/' +
                    clf_key + '/' + clf_key + '.pkl')'''
            clf_key =\
                'edxclassify/saved_clf' + '/' +\
                    clf_key + '/' + clf_key + '.pkl'
        print clf_key
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
        data = []
        if self.name in DEFAULT_CLF:
            for e in examples:
                example_arr = [''] * len(FEATURE_COLUMNS)
                for k in e.keys():
                    example_arr[FEATURE_COLUMNS[k]] = e[k]
                data.append(example_arr)
        else:
            data = examples
        X = self.dc.process_records_without_labels(data)
        return self.clf.test(X, None)
