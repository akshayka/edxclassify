from abstract_data_cleaner import DataCleaner
import dc_util
from edx import Edx


class EdxSentiment(Edx):
    def __init__(self,
                 binary=False,
                 collapse_numbers=False,
                 extract_noun_phrases=False,
                 upweight_first_sentence=False):
        super(EdxSentiment, self).__init__(binary, collapse_numbers,
                                           extract_noun_phrases,
                                           upweight_first_sentence)
        self.name = 'EdxSentiment'

    def labels(self):
        if self.binary:
            return ['negative', 'positive']
        else:
            return ['negative', 'neutral', 'positive']

    def process_doc(self, document):
        return super(EdxSentiment, self).process_document(document)

    # The first entry in each record is the document;
    # the fifth entry in each record is the sentiment likert score.
    def process_records(self, records):
        return [(self.process_doc(record[0]),
                dc_util.compress_likert(int(float(record[4])), self.binary, 3))
                for record in records]
