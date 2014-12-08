import re
from custom_stop_words import ENGLISH_STOP_WORDS
from classify.feature_spec import FEATURE_COLUMNS


class TextCurator:
    def __init__(self, token_pattern,
                 stop_words='english',
                 binary_counts=False):
        self.token_pattern = token_pattern
        self.binary_counts = binary_counts

        if stop_words == 'english':
            stop_words = ENGLISH_STOP_WORDS
        self.stop_words = stop_words
        self.keyset = None

    def fit(self, documents, y=None):
        self.keyset = set(
            [token for document in documents
             for token in re.findall(self.token_pattern, document)]
        ) - self.stop_words

    def transform(self, documents, y=None):
        result = []
        for document in documents:
            countDict = dict.fromkeys(self.keyset, 0)
            for token in re.findall(self.token_pattern, document):
                if token in countDict:
                    if self.binary_counts:
                        countDict[token] = 1
                    else:
                        countDict[token] += 1
            result = [result, countDict]
        return result

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


class FeatureExtractor:
    def __init__(self, feature_name):
        self.feature_name = feature_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        idx = FEATURE_COLUMNS[self.feature_name]
        return [row[idx] for row in X]


class FeatureCurator:
    def __init__(self, feature_name, curate_function):
        self.feature_name = feature_name
        self.curate = curate_function

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [{self.feature_name + ' feature': self.curate(value)}
                for value in X]

    def fit_transform(self, X, y=None):
        return self.transform(X)


class FeatureAggregator:
    def __init__(self, feature_curators):
        self.feature_curators = feature_curators

    def extractFeature(self, X, feature_name):
        idx = FEATURE_COLUMNS[feature_name]
        return [row[idx] for row in X]

    def fit_transform(self, X, y=None):
        '''
        Aggregate each of the features on a per-row basis, so that transform
        can return a list of dictionaries, one per training example
        '''
        result = None
        for name, curator in self.feature_curators:
            features = self.extractFeature(X, name)
            dicts = curator.fit_transform(features)
            if result == None:
                result = dicts
            else:
                map(dict.update, result, dicts)
        return result

        # Features to include besides text:
        # type, anonymous_to_peers, up_count, reads, cum_attempts,
        # cum_points, points_possible

