from abstract_data_cleaner import DataCleaner
import dc_util


class EdxUrgency(DataCleaner):
    def __init__(self, tfidf=False):
        self.name = 'EdxUrgency'
        
    def labels(self):
        return ['noise', 'neutral', 'urgent']

    # The first entry in each record is the document;
    # the seventh entry in each record is the likert score.
    def process_records(self, records):
        return [(record[0], dc_util.compress_likert(int(float(record[6])))) \
                for record in records]
