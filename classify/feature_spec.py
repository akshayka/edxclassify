""" Schema of goldsets

    Variables tagged with 'unused' are not used by SKLearnCLF
"""

FEATURE_COLUMNS = {
    'text': 0,                  # The body of the forum post
    'opinion': 1,               # Indicator in {0, 1}, 1 iff post is an opinion
    'question': 2,              # Indicator in {0, 1}, 1 iff post is a question
    'answer': 3,                # Indicator in {0, 1}, 1 iff post is an answer
    'sentiment': 4,             # Likert in [1, 7]
    'confusion': 5,             # Likert in [1, 7]
    'urgency': 6,               # Likert in [1, 7]
    'poster_identifiable': 7,   # Indicator in {0, 1}, 1 iff identifiable
                                # (unused)
    'course_type': 8,           # string descriptor of containing course
                                # (unused)
    'forum_pid': 9,             # (unused) 
    'course_name': 10,          # (unused)
    'forum_uid': 11,            # (unused)
    'date': 12,                 # date posted (unused)
    'post_type': 13,            # Indicator in {CommentThread, Comment},
                                # CommentThread iff post started a thread
    'anonymous': 14,            # "False" iff not anonymous
    'anonymous_to_peers': 15,   # "False" iff not anonymous to peers
    'up_count': 16,             # Number of upvotes received by post
    'comment_thread_id': 17,    # (unused)
    'reads': 18,                # Number of reads garnered by thread
    'cum_attempts': 19,         # Cumulative number of attempts made by poster
                                # (unused)
    'cum_grade': 20             # Cumulative grade of poster
                                # (unused)
}
