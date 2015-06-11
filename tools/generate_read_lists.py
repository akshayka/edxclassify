import argparse
import cPickle

parser = argparse.ArgumentParser(
                  description='make lists to feed to hist maker')
parser.add_argument('goldfile', type=str, help='gold file -- the dataset')
args = parser.parse_args()

try:
    with open(args.goldfile, 'rU') as goldfile:
        goldlist = cPickle.load(goldfile)
        course_names = set([record[10] for record in goldlist if record[10] != ''])
        outfiles = { name : open(name.replace('/', '_') + '_reads', 'wb')\
                        for name in course_names }
        reads_per_course = {}
        for record in goldlist:
            name = record[10]
            if name == '':
                continue
            reads_per_course.setdefault(name, []).append(record[18])
        for name in course_names:
            cPickle.dump(reads_per_course[name], outfiles[name])
except IOError:
    print 'failed to open file'
