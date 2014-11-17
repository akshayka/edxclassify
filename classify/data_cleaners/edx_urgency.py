from abstract_data_cleaner import DataCleaner
import dc_util
from edx import Edx


class EdxUrgency(Edx):
    def __init__(self,
                 binary=False,
                 collapse_numbers=False,
                 extract_noun_phrases=False,
                 upweight_first_sentence=False):
        super(EdxUrgency, self).__init__(binary, collapse_numbers,
                                         extract_noun_phrases,
                                         upweight_first_sentence)
        self.name = 'EdxUrgency'

    def labels(self):
        if self.binary:
            return ['non-urgent', 'urgent'] 
        else:
            return ['non-urgent', 'neutral', 'urgent']

    def process_doc(self, document):
        return super(EdxSentiment, self).process_document(document)

    # The first entry in each record is the document;
    # the seventh entry in each record is the urgency likert score.
    def process_records(self, records):
        return [(record[0], dc_util.compress_likert(int(float(record[6])),
                self.binary, 4)) for record in records]
