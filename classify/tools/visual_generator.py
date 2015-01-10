from classify.tools.configurations import *
import classify.harness as harness
import numpy as np
import sys
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib
import pandas
import seaborn as sns
from across_courses import HUMANITIES_COURSES, MEDICINE_COURSES


def make_f1_table(results, output_file):
    stdout_save = sys.stdout
    sys.stdout = open(output_file, "wb")
    print "Test Name, Train f1 Score, Test f1 Score"
    for name in results:
        train = extract_f1_train_score(results[name])
        test = extract_f1_test_score(results[name])
        print name + ", " + str(train) + ", " + str(test)
    sys.stdout.close()
    sys.stdout = stdout_save



def make_word_cloud(command, data_path, output_file):
    harness_args = [data_path, "edx_urgency"]
    harness_args = harness_args + command.split(' ')

    x, y, words = harness.main(harness_args)
    stdout_save = sys.stdout
    sys.stdout = open(output_file, 'wb')
    for w in words['knowledgeable']:
        for a in w:
            for b in a:
                print b.replace("text_document__", "")
    sys.stdout.close()
    sys.stdout = stdout_save


def evaluate_configurations(data_path, CONFIGURATIONS):
    try:
        completed = pickle.load(open('visual/completed_evaluations.p', 'r'))
        print "Successfully loaded previous results"
        for command in completed:
            print "-> Loaded command: %s" % command
    except EOFError:
        completed = {}

    result = {}
    for name, command in CONFIGURATIONS:
        if command in completed:
            print "Loading previously completed results for command: %s" \
                  % command
            result[name] = completed[command]
        else:
            harness_args = [data_path, "edx_confusion"]
            harness_args = harness_args + command.split(' ')
            result[name] = tuple(harness.main(harness_args))
            completed[command] = result[name]
            pickle.dump(completed, open('visual/completed_evaluations.p', 'w'))
            print "Added command to cached results: %s" % command

    return result

# Structure of results dictionary:
# { "config-name":
#   (
#       [precision_train, recall_train, f1_train],
#       [precision_test,  recall_test,  f1_test ],
#       relevant_features
#   )
# }
# *_train and *_test are lists. relevant_features is a dictionary
# with keys 'knowledgeable', 'neutral', and 'confused'. The values
# are lists of lists of the meaningful tokens.


def extract_f1_test_score(results_dict_entry):
    """ Extract the average f1 test score from the given entry.

    The results_dict_entry should be one of the values in the
    results_dictionary returned by evaluate_configurations.
    """
    f1_sum = 0
    for f1_tuple in results_dict_entry[1][2]:
        f1_sum += f1_tuple[2]
    return f1_sum / len(results_dict_entry[1][2])

def extract_f1_train_score(results_dict_entry):
    f1_sum = 0
    for f1_tuple in results_dict_entry[0][2]:
        f1_sum += f1_tuple[2]
    return f1_sum / len(results_dict_entry[0][2])


def extract_precision_test_score(results_dict_entry):
    p_sum = 0
    for p_tuple in results_dict_entry[1][0]:
        p_sum += p_tuple[2]
    return p_sum / len(results_dict_entry[1][0])


def extract_recall_test_score(results_dict_entry):
    r_sum = 0
    for r_tuple in results_dict_entry[1][1]:
        r_sum += r_tuple[2]
    return r_sum / len(results_dict_entry[1][1])


def make_score_graph(results_dictionary, output_filename, plot_title):
    labels = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for name in results_dictionary:
        labels.append(name)
        entry = results_dictionary[name]
        f1_scores.append(extract_f1_test_score(entry))
        precision_scores.append(extract_precision_test_score(entry))
        recall_scores.append(extract_recall_test_score(entry))
    x = np.array(["f1 score", "precision", "recall"])
    metric_labels = np.repeat(x, [len(f1_scores)] * 3)
    data = pandas.DataFrame(
        {"Classifier": pandas.Categorical(labels * 3),
         "Performance": pandas.Series(f1_scores + precision_scores + recall_scores),
         "Metric": pandas.Series(metric_labels)}
    )
    bargraph_helper("Classifier", "Performance", "Metric",
                    data, plot_title, output_filename)

def bargraph_helper(l1, l2, l3, data, plot_title, output_filename):
    sns.set(style="white", context="poster")
    fig, ax = plt.subplots(1, 1)
    sns.barplot(l1, l2, l3, data)
    plt.xticks(rotation=15, ha='right')
    plt.title(plot_title)
    # plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(output_filename)

def make_class_graph():
    bargraph_helper("Course Title", "Performance", "Metric",
                    HUMANITIES_COURSES, "Cross-Validation with Humanities Courses", "across-hum.png")

    bargraph_helper("Course Title", "Performance", "Metric",
                    MEDICINE_COURSES, "Cross-Validation with Courses in Medicine", "across-med.png")


def main(args=None):
    data_path = "data/gold_sets/humanities_gold_v8"
    if len(args) > 1:
        data_path = args[1]
    make_word_cloud("logistic -t 5 -c -n -p 0.4 -l -url -txt", data_path, "visual/word_cloud.txt")
    #all_results = evaluate_configurations(data_path, ALL_CONFIGURATIONS)
    #default_results = evaluate_configurations(data_path, DEFAULT_CONFIGURATIONS)
    #tfidf_results = evaluate_configurations(data_path, TFIDF_CONFIGURATIONS)
    #kbest_results = evaluate_configurations(data_path, KBEST_CONFIGURATIONS)

    #make_score_graph(default_results, 'default-results-bar.png', 'Classification without Feature Reduction')
    #make_score_graph(tfidf_results, 'tfidf-results-bar.png', 'Classification with TF-IDF')
    #make_score_graph(kbest_results, 'kbest-results-bar.png', 'Classification with K-Best Features')
    # make_class_graph()

    #make_f1_table(all_results, "visual/classifier_results.csv")

if __name__ == '__main__':
    main(sys.argv)
