from abstract_data_cleaner import DataCleaner


class EdxSentiment(DataCleaner):
    # The first entry in each record is the document;
    # the fifth entry in each record is the likert score.
    def process_records(records):
        return [(record[0], record[4]) for record in records]
