from abstract_data_cleaner import DataCleaner
import dc_util
from edx import Edx


class EdxSentiment(Edx):
    def __init__(self,
                 binary=False,
                 collapse_numbers=False,
                 latex=True,
                 extract_noun_phrases=False,
                 first_sentence_weight=1):
        super(EdxSentiment, self).__init__(binary, collapse_numbers, latex,
                                           extract_noun_phrases,
                                           first_sentence_weight)
        self.name = 'EdxSentiment ' + self.name

    def labels(self):
        if self.binary:
            return ['negative', 'positive']
        else:
            return ['negative', 'neutral', 'positive']

    def process_doc(self, document):
        return super(EdxSentiment, self).process_doc(document)

    # The first entry in each record is the document;
    # the fifth entry in each record is the sentiment likert score.
    def process_records(self, records):
        return [(self.process_doc(record[0]),
                dc_util.compress_likert(int(float(record[4])), self.binary, 3))
                for record in records]
