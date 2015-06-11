SMALL_CONFIGURATIONS = [
    ("Logistic: Text Only",            "logistic -t 5 -c -n -p 0.4 -txt"        ),
    ("Logistic: All Features",         "logistic -t 5 -c -n -p 0.4"             )
 ]

ALL_CONFIGURATIONS = [
    ("Logistic: Text Only",            "logistic -t 5 -c -n -p 0.4 -txt"        ),
    ("Logistic: All Features",         "logistic -t 5 -c -n -p 0.4"             ),
    ("Logistic: Tf-Idf, Text Only",    "logistic -t 5 -c -n -p 0.4 -tf -txt"    ),
    ("Logistic: Tf-Idf, All Features", "logistic -t 5 -c -n -p 0.4 -tf"         ),
    ("Logistic: K-Best, Text Only",    "logistic -t 5 -c -n -p 0.4 -kb 200 -txt"),
    ("Logistic: K-Best, All Features", "logistic -t 5 -c -n -p 0.4 -kb 200"     ),
    ("SVM: Text Only",                 "lin_svc -t 5 -c -n -p 0.0625 -txt"         ),
    ("SVM: All Features",              "lin_svc -t 5 -c -n -p 0.0625"              ),
    ("SVM: Tf-Idf, Text Only",         "lin_svc -t 5 -c -n -p 0.0625 -tf -txt"     ),
    ("SVM: Tf-Idf, All Features",      "lin_svc -t 5 -c -n -p 0.0625 -tf"          ),
    ("SVM: K-Best, Text Only",         "lin_svc -t 5 -c -n -p 0.0625 -kb 200 -txt" ),
    ("SVM: K-Best, All Features",      "lin_svc -t 5 -c -n -p 0.0625 -kb 200"      ),
    ("Mult. Bayes: Text Only",            "naive_bayes -t 5 -c -n -p 0.4 -txt"        ),
    ("Mult. Bayes: All Features",         "naive_bayes -t 5 -c -n -p 0.4"             ),
    ("Mult. Bayes: Tf-Idf, Text Only",    "naive_bayes -t 5 -c -n -p 0.4 -tf -txt"    ),
    ("Mult. Bayes: Tf-Idf, All Features", "naive_bayes -t 5 -c -n -p 0.4 -tf"         ),
    ("Mult. Bayes: K-Best, Text Only",    "naive_bayes -t 5 -c -n -p 0.4 -kb 200 -txt"),
    ("Mult. Bayes: K-Best, All Features", "naive_bayes -t 5 -c -n -p 0.4 -kb 200"     )
    ]

DEFAULT_CONFIGURATIONS = [
    ("Logistic: Text Only",            "logistic -t 5 -c -n -p 0.4 -txt"        ),
    ("Logistic: All Features",         "logistic -t 5 -c -n -p 0.4"             ),
    ("SVM: Text Only",                 "lin_svc -t 5 -c -n -p 0.0625 -txt"         ),
    ("SVM: All Features",              "lin_svc -t 5 -c -n -p 0.0625"              ),
    ("Mult. Bayes: Text Only",         "naive_bayes -t 5 -c -n -p 0.4 -txt"     ),
    ("Mult. Bayes: All Features",      "naive_bayes -t 5 -c -n -p 0.4"          )
    ]

TFIDF_CONFIGURATIONS = [
    ("Logistic: Text Only",    "logistic -t 5 -c -n -p 0.4 -tf -txt"    ),
    ("Logistic: All Features", "logistic -t 5 -c -n -p 0.4 -tf"         ),
    ("SVM: Text Only",         "lin_svc -t 5 -c -n -p 0.0625 -tf -txt"     ),
    ("SVM: All Features",      "lin_svc -t 5 -c -n -p 0.0625 -tf"          ),
    ("Mult. Bayes: Text Only",    "naive_bayes -t 5 -c -n -p 0.4 -tf -txt"    ),
    ("Mult. Bayes: All Features", "naive_bayes -t 5 -c -n -p 0.4 -tf"         ),
    ]

KBEST_CONFIGURATIONS = [
    ("Logistic: Text Only",    "logistic -t 5 -c -n -p 0.4 -kb 200 -txt"),
    ("Logistic: All Features", "logistic -t 5 -c -n -p 0.4 -kb 200"     ),
    ("SVM: Text Only",         "lin_svc -t 5 -c -n -p 0.0625 -kb 200 -txt" ),
    ("SVM: All Features",      "lin_svc -t 5 -c -n -p 0.0625 -kb 200"      ),
    ("Mult. Bayes: Text Only",    "naive_bayes -t 5 -c -n -p 0.4 -kb 200 -txt"),
    ("Mult. Bayes: All Features", "naive_bayes -t 5 -c -n -p 0.4 -kb 200"     )
    ]
