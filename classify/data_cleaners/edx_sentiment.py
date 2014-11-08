from abstract_data_cleaner import DataCleaner
import dc_util


class EdxSentiment(DataCleaner):
    def __init__(self, tfidf=False):
        self.name = 'EdxSentiment'
        
    def labels(self):
        return ['negative', 'neutral', 'positive']

    # The first entry in each record is the document;
    # the fifth entry in each record is the likert score.
    def process_records(self, records):
        return [(record[0], dc_util.compress_likert(int(float(record[4])))) \
                for record in records]
