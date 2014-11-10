from abstract_data_cleaner import DataCleaner
import dc_util


class EdxUrgency(DataCleaner):
    def __init__(self, binary=False):
        self.name = 'EdxUrgency'
        self.binary = binary
        
    def labels(self):
        if self.binary:
            return ['non-urgent', 'urgent'] 
        else:
            return ['non-urgent', 'neutral', 'urgent']

    # The first entry in each record is the document;
    # the seventh entry in each record is the likert score.
    def process_records(self, records):
        return [(record[0], dc_util.compress_likert(int(float(record[6])),
                self.binary, 4)) for record in records]
