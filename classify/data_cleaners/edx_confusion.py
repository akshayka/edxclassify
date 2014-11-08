from abstract_data_cleaner import DataCleaner
import dc_util


class EdxConfusion(DataCleaner):
    def __init__(self, tfidf=False):
        self.name = 'EdxConfusion'
        
    def labels(self):
        return ['knowledgeable', 'neutral', 'confused']

    # The first entry in each record is the document;
    # the sixth entry in each record is the likert score.
    def process_records(self, records):
        return [(record[0], dc_util.compress_likert(int(float(record[5])))) \
                for record in records]
