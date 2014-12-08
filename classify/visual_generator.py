import classify.harness as harness
import numpy as np
import matplotlib.pyplot as plt

data_path = "data/gold_sets/medicine_gold_v8"

CONFIGURATIONS = [("best-logistic", "logistic -t 5 -c -n -p 0.4")]

def evaluate_configurations():
    result = {}
    for name, config in CONFIGURATIONS:
        harness_args = [data_path, "edx_confusion"]
        harness_args = harness_args + config.split(' ')
        result[name] = tuple(harness.main(harness_args))
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
    return np.mean(results_dict_entry[1][2])


def extract_precision_test_score(results_dict_entry):
    return np.mean(results_dict_entry[1][0])

def extract_recall_test_score(results_dict_entry):
    return np.mean(results_dict_entry[1][1])


def make_score_graph(results_dictionary):
    labels = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for name, entry in results_dictionary:
        labels.append(name)
        f1_scores.append(extract_f1_test_score(entry))
        precision_scores.append(extract_precision_test_score(entry))
        recall_scores.append(extract_recall_test_score(entry))
    x = xrange(len(scores))
    bar_width = 0.25
    fig, ax = plt.subplots()
    plt.bar(x, f1_scores, bar_width, color='g', label='f1 score')
    plt.bar(x, precision_scores, bar_width, color='r', label='precision')
    plt.bar(x, recall_scores, bar_width, color='b', label='recall')
    plt.xlabel('Classifier')
    plt.ylabel('Performance')
    plt.xticks(x + bar_width, labels, rotation=40, ha='right')
    plt.show()


def main(args=None):
    results_dictionary = evaluate_configurations()
    make_f1_score_graph(results_dictionary)


if __name__ == 'main':
    main()
