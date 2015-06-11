import cPickle
from classify.feature_spec import FEATURE_COLUMNS

with open(raw_input('Enter data file: '), 'rb') as infile:
    dataset = cPickle.load(infile)[1:]
    for record in dataset:
        if float(record[FEATURE_COLUMNS['urgency']]) > 4:
            print record[0]
            print record[FEATURE_COLUMNS['urgency']]
            raw_input('Press Enter to continue ...')

