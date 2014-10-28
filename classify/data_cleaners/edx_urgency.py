from abstract_data_cleaner import DataCleaner


class EdxUrgency(DataCleaner):
    # The first entry in each record is the document;
    # the seventh entry in each record is the likert score.
    def process_records(self, records):
        return [(record[0], int(float(record[6]))) for record in records]
