from abstract_data_cleaner import DataCleaner
import dc_util
from edx import Edx


class EdxUrgency(Edx):
    def __init__(self,
                 binary=False,
                 extract_noun_phrases=False,
                 first_sentence_weight=1):
        super(EdxUrgency, self).__init__(binary,
                                         extract_noun_phrases,
                                         first_sentence_weight)
        self.name = 'EdxUrgency ' + self.name

    def labels(self):
        if self.binary:
            return ['non-urgent', 'urgent'] 
        else:
            return ['non-urgent', 'neutral', 'urgent']

    def process_doc(self, document):
        return super(EdxUrgency, self).process_doc(document)

    # TUPLE(LIST<features>, label)
    def process_records(self, records):
        return [([self.process_doc(record[self.columns['text']])] +\
                record[1:],
                dc_util.compress_likert(record[self.columns['urgency']],
                                        self.binary, 4))
                for record in records]
