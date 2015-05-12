from sklearn.externals import joblib

def load_clf(pkl_file):
    data_cleaner, clf = joblib.load(pkl_file)
    return data_cleaner, clf
