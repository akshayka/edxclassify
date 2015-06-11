import cPickle
from classify.feature_spec import FEATURE_COLUMNS
import random

goldfile = raw_input('Enter goldfile:  ')
records = cPickle.load(open(goldfile, 'rb'))
not_confused = []
balanced = []

for record in records:
    if float(record[FEATURE_COLUMNS['confusion']]) > 4:
        balanced.append(record)
    else:
        not_confused.append(record)

print len(balanced)
print len(not_confused)
indices = random.sample(xrange(len(not_confused)), len(balanced))

for i in indices:
    balanced.append(not_confused[i])

print len(balanced)
cPickle.dump(balanced, open(raw_input('outfile: '), 'wb'))
