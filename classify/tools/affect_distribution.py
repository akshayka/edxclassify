"""
Summarize affect distributions at varying course granularities.
"""

from classify.feature_spec import FEATURE_COLUMNS
from classify.data_cleaners.dc_util import compress_likert
import cPickle
import glob
import os
import sys

def post_fractions(posts):
    top_posters = sorted(posts.items(), key=lambda (k, v): v)[-10:]
    total_count = sum(posts.values())
    output = []
    for i in range(len(top_posters)):
        output.append((top_posters[i][0][:3],
                    '%.2f%%' % (100 * float(top_posters[i][1]) / total_count)))
                        
    #post_fractions = [100 * float(c) / total_count for c in post_counts[-20:]]
    #post_fractions = ['%.2f%%' % f for f in post_fractions]
    return str(output)

def percent_posts(posts, total):
    return '%.2f%%' % (sum(posts.values()) / total * 100)

print sys.argv[1]
file_paths = [f for f in glob.glob(sys.argv[1]) if os.path.isfile(f)]
uid_pos = FEATURE_COLUMNS['forum_uid']
sent_pos = FEATURE_COLUMNS['sentiment']
conf_pos = FEATURE_COLUMNS['confusion']
urg_pos = FEATURE_COLUMNS['urgency']
op_pos = FEATURE_COLUMNS['opinion']
ans_pos = FEATURE_COLUMNS['answer']
q_pos = FEATURE_COLUMNS['question']
for path in file_paths:
    with open(path, 'rb') as f:
        dataset = cPickle.load(f)
        # dictionaries of persons --> affect
        posts = {}
        sentiment = [{}, {}, {}]
        confusion = [{}, {}, {}]
        urgency = [{}, {}, {}]
        opinion = [{}, {}]
        answer = [{}, {}]
        question = [{}, {}]
        for record in dataset:
            if len(record) > uid_pos and record[uid_pos] != '':
                forum_uid = record[uid_pos]
                posts[forum_uid] = posts.get(forum_uid, 0) + 1

                s = compress_likert(record[sent_pos])
                sentiment[s][forum_uid] = sentiment[s].get(forum_uid, 0) + 1

                c = compress_likert(record[conf_pos])
                confusion[c][forum_uid] = confusion[c].get(forum_uid, 0) + 1

                u = compress_likert(record[urg_pos])
                urgency[u][forum_uid] = urgency[u].get(forum_uid, 0) + 1

                o = int(record[op_pos])
                opinion[o][forum_uid] = opinion[o].get(forum_uid, 0) + 1

                a = int(record[ans_pos])
                answer[a][forum_uid] = answer[a].get(forum_uid, 0) + 1

                q = int(record[q_pos])
                question[q][forum_uid] = question[q].get(forum_uid, 0) + 1

        # number of persons in dictionary
        # percentage of posts accounted for by top 10 posters
        # percentage of affect posts accounted for by top 10 posters
        total = float(sum(posts.values()))
        print '------- Course: %s --------' % os.path.basename(path)
        print '\tTotal posts: %d' % total
        print '\tDistinct posters: %d' % len(posts)
        print '\t  ' + post_fractions(posts)
        print '\tSentiment:'
        print '\t  Negative: ' + percent_posts(sentiment[0], total)
        print '\t  ' + post_fractions(sentiment[0])
        print '\t  Neutral: ' + percent_posts(sentiment[1], total)
        print '\t  ' + post_fractions(sentiment[1])
        print '\t  Positive: ' + percent_posts(sentiment[2], total)
        print '\t  ' + post_fractions(sentiment[2])
        print '\tConfusion'
        print '\t  Knowledgeable: ' + percent_posts(confusion[0], total)
        print '\t  ' + post_fractions(confusion[0])
        print '\t  Neutral: ' + percent_posts(confusion[1], total)
        print '\t  ' + post_fractions(confusion[1])
        print '\t  Confused: ' + percent_posts(confusion[2], total)
        print '\t  ' + post_fractions(confusion[2])
        print '\tUrgency'
        print '\t  Non-urgent: ' + percent_posts(urgency[0], total)
        print '\t  ' + post_fractions(urgency[0])
        print '\t  Semi-urgent: ' + percent_posts(urgency[1], total)
        print '\t  ' + post_fractions(urgency[1])
        print '\t  Urgent: ' + percent_posts(urgency[2], total)
        print '\t  ' + post_fractions(urgency[2])
        print '\tOpinion'
        print '\t  Fact: ' + percent_posts(opinion[0], total)
        print '\t  ' + post_fractions(opinion[0])
        print '\t  Opinion: ' + percent_posts(opinion[1], total)
        print '\t  ' + post_fractions(opinion[1])
        print '\tAnswer'
        print '\t  Not Answer: ' + percent_posts(answer[0], total)
        print '\t  ' + post_fractions(answer[0])
        print '\t  Answer: ' + percent_posts(answer[1], total)
        print '\t  ' + post_fractions(answer[1])
        print '\tQuestion'
        print '\t  Not Question: ' + percent_posts(question[0], total)
        print '\t  ' + post_fractions(question[0])
        print '\t  Question: ' + percent_posts(question[1], total)
        print '\t  ' + post_fractions(question[1])
