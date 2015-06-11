import edxclassify.harness as harness
import glob
import os
import sys

classes = ['question', 'answer', 'opinion', 'confusion', 'sentiment', 'urgency']
cmd = 'logistic -p 0.27 -c -b -wl'

print sys.argv[1]
file_paths = [f for f in glob.glob(sys.argv[1]) if os.path.isfile(f)]
for f in file_paths:
    for c in classes:
        output_file = 'data/wordlists/' + os.path.basename(f) + '_' + c
        harness_args = [f, c] + cmd.split(' ') + [output_file]
        if os.path.isfile(output_file):
            print 'found results for ' + ' '.join(harness_args) + '; skipping.'
        else:
            print 'executing ' + ' '.join(harness_args) + ' ...'
            harness.main(harness_args)

