from abstract_data_cleaner import DataCleaner


class EdxConfusion(DataCleaner):
    # The first entry in each record is the document;
    # the sixth entry in each record is the likert score.
    def process_records(self, records):
        return [(record[0], int(float(record[5]))) for record in records]
