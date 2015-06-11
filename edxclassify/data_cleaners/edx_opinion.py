from abstract_data_cleaner import DataCleaner
import dc_util
from edx import Edx


class EdxOpinion(Edx):
    def __init__(self,
                 extract_noun_phrases=False,
                 first_sentence_weight=1):
        super(EdxOpinion, self).__init__(True,
                                         extract_noun_phrases,
                                         first_sentence_weight)
        self.name = 'EdxOpinion ' + self.name

    def labels(self):
        return ['not-opinion', 'opinion']

    def process_doc(self, document):
        return super(EdxOpinion, self).process_doc(document)

    # LIST(TUPLE(LIST(features), label))
    def process_records(self, records):
        return [([self.process_doc(record[self.columns['text']])] +\
                record[1:], int(record[self.columns['opinion']]))\
                for record in records]
