import pickle
from data_cleaner_factory import make_data_cleaner

with open('../data/gold_sets/math_gold', 'rb') as infile:
    dataset = pickle.load(infile)[1:]
    for record in dataset:
        if '$' in record:
        #if (record[5] > 4):
            print record[0]
            raw_input('Press Enter to continue ...')
