from abstract_data_cleaner import DataCleaner
import dc_util


class EdxSentiment(DataCleaner):
    def __init__(self, binary=False):
        self.name = 'EdxSentiment'
        self.binary = binary
        
    def labels(self):
        if self.binary:
            return ['negative', 'non-negative']
        else:
            return ['negative', 'neutral', 'positive']

    # The first entry in each record is the document;
    # the fifth entry in each record is the likert score.
    def process_records(self, records):
        return [(record[0], dc_util.compress_likert(int(float(record[4])),
                 self.binary, 3)) for record in records]
