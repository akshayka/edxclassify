from abstract_data_cleaner import DataCleaner
import dc_util
from edx import Edx


class EdxConfusion(Edx):
    def __init__(self,
                 binary=False,
                 extract_noun_phrases=False,
                 first_sentence_weight=1):
        super(EdxConfusion, self).__init__(binary,
                                           extract_noun_phrases,
                                           first_sentence_weight)
        self.name = 'EdxConfusion ' + self.name

    def labels(self):
        if self.binary:
            return ['not-confused', 'confused']
        else:
            return ['knowledgeable', 'neutral', 'confused']

    def process_doc(self, document):
        return super(EdxConfusion, self).process_doc(document)

    # LIST(TUPLE(LIST(features), label))
    def process_records(self, records):
        return [([self.process_doc(record[self.columns['text']])] +\
                record[1:],
                dc_util.compress_likert(record[self.columns['confusion']],
                                        self.binary, 4))
                for record in records]

    def process_records_without_labels(self, records):
        return [ [self.process_doc(record[self.columns['text']])] +\
                  record[1:] for record in records ]
