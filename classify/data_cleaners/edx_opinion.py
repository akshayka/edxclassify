from abstract_data_cleaner import DataCleaner
import dc_util
from edx import Edx


class EdxOpinion(Edx):
    def __init__(self,
                 collapse_numbers=False,
                 extract_noun_phrases=False,
                 first_sentence_weight=1):
        super(EdxOpinion, self).__init__(True, collapse_numbers,
                                           extract_noun_phrases,
                                           first_sentence_weight)
        self.name = 'EdxOpinion ' + self.name

    def labels(self):
        return ['not-opinion', 'opinion']

    def process_doc(self, document):
        return super(EdxOpinion, self).process_doc(document)

    # The first entry in each record is the document;
    # the second entry in each record is the sentiment likert score.
    def process_records(self, records):
        return [(self.process_doc(record[0]),
                dc_util.compress_likert(int(float(record[1])), True, 0))
                for record in records]
