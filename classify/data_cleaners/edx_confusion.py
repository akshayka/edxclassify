from abstract_data_cleaner import DataCleaner
import dc_util


class EdxConfusion(DataCleaner):
    def __init__(self, binary=False):
        self.name = 'EdxConfusion'
        self.binary = binary
        
    def labels(self):
        if self.binary:
            return ['not-confused', 'confused']
        else:
            return ['knowledgeable', 'neutral', 'confused']

    # The first entry in each record is the document;
    # the sixth entry in each record is the likert score.
    def process_records(self, records):
        return [(record[0], dc_util.compress_likert(int(float(record[5])),
                self.binary, 4)) for record in records]
