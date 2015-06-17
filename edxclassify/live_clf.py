"""
edxclassify.live_clf

This module contains the LiveCLF class, a wrapper around classifiers
that were trained and serialized to disk.

The edxclassify package comes coupled with a number of pre-trained
classifiers, the names of which are in the DEFAULT_CLF list. LiveCLF
can load any classifier that was serialized using edxclassify.harness,
and in particular it can load these pre-trained classifiers as well.

If a default classifier is used, then each example fed to its wrapper
LiveCLF must adhere to the schema laid out in EXAMPLE_FMT.

For example, the following loads a default classifier for confusion trained on
technical courses:

    clf = LiveCLF('confusion_technical')

We can then proceed to classify posts by feeding them to LiveCLF's predict
method, which takes an array of examples. Since we're using a default
classifier, each example must be a dictionary whose keyset is exactly equal to
that of EXAMPLE_FMT. For example:

clf.predict([ {'text': Help me, I am so confused.'\
                       ' Can someone please explain this problem to me?',
               'post_type': 'CommentThread',
               'anonymous': 'True',
               'anonymous_to_peers': 'True',
               'up_count': 50,
               'reads': 200,}]

will output an array containing a 0/1 prediction for each example, 0 being
not confused and 1 being confused. In this case, the array will be of length
one.

LiveCLF can also be used to load arbitrary classifiers that the user herself
has serialized to disk using edxclassify.harness. For such a classifier, 
each example must be a _list_ corresponding to the input expected by its
respective data cleaner.

To bypass LiveCLF and retrieve classifiers directly, users may use
edxclassify.clf_util.load_clf.  
"""

import edxclassify.classifiers.clf_util as clf_util
from edxclassify.feature_spec import FEATURE_COLUMNS
import pkg_resources

EXAMPLE_FMT = {
    'text': '',                 # (String) The body of the forum post
    'post_type': '',            # (String) Indicator in {CommentThread, Comment},
                                #         CommentThread iff post started a thread
    'anonymous': '',            # (String) "False" iff not anonymous
    'anonymous_to_peers': '',   # (String) "False" iff not anonymous to peers
    'up_count': 0,              # (Int)     Number of upvotes received by post
    'reads': 0,                 # (Int)     Number of reads garnered by thread
}

EXAMPLE_KEYS = frozenset(EXAMPLE_FMT.keys())

DEFAULT_CLF = frozenset([
    """
    This list contains all the pre-trained classifiers available
    to users. Keys without suffixes correspond to classifiers that were
    trained on a gold set containing the entirety of the MOOCPosts Dataset.
    Those with the suffix _stats correspond to classifiers trained on the four
    statistics courses; those with _econ correspond to classifiers trained on
    the two economics courses; those with _technical correspond to classifiers
    trained on technical courses (i.e., STEM); etc.

    Each classifier was built using a binary logistic regression model,
    supplemented by subclassifiers (-c in harness), with l2-regularization
    of 0.27.
    """

    'confusion_technical',
    'confusion_nontechnical',
])

class LiveCLF:
    """
    A wrapper class that loads previously serialized classifiers and provides
    an interface to them.

    parameters:
    ----------
    clf_key : String - Determines which classifier is loaded.
                       If clf_key is an element of DEFAULT_CLF, then the
                       corresponding packaged classifier will be loaded.

                       Otherwise, clf_key is interpreted as a path to a .pkl file
                       that resides in the directory to which the classifier was saved.

    attributes:
    ----------
    name : String            - the clf_key
    dc : DataCleaner - The data cleaner with which the
                               classifier was coupld
    clf : Classifier - The underlying classifier. Default classifiers
                       are all SklearnCLF's
                       (edxclassify.classifiers.sklearn_clf.SklearnCLF)
    """
    def __init__(self, clf_key):
        """
        """ 
        self.name = clf_key
        if clf_key in DEFAULT_CLF:
            clf_key = pkg_resources.resource_filename(
                'edxclassify', 'saved_clf' + '/' +
                    clf_key + '/' + clf_key + '.pkl')
        print clf_key
        dc, clf = clf_util.load_clf(clf_key)
        self.dc = dc
        self.clf = clf

    def predict(self, examples):
        """Generate predictions for examples using underlying classifier

        parameters:
        ----------
        examples - either a list of dictionaries (one per example) each laid out
                   as per EXAMPLE_FMT, if clf.name is a default classifier,
                   or a list of arrays (one per example) otherwise.
        
        returns:
        --------
        an array of shape [n_examples], with one prediction for each example
        """
        data = []
        if self.name in DEFAULT_CLF:
            for e in examples:
                ekeys = set(e.keys())
                assert ekeys == EXAMPLE_KEYS,\
                    'When using default a classifier, each example '\
                    'must contain the exact keys found in EXAMPLE_FMT'
                example_arr = [''] * len(FEATURE_COLUMNS)
                for k in e.keys():
                    example_arr[FEATURE_COLUMNS[k]] = e[k]
                data.append(example_arr)
        else:
            data = examples
        X = self.dc.process_records_without_labels(data)
        return self.clf.test(X, None)
