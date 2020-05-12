import argparse
import itertools
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.problem_transform import ClassifierChain

from mlutils import clean_df, tokenize_lemmas, tokenize_nounphrases, tokenize_nouns

RANDOM_SEED = 56
N_CORES = 20

try:
    data_dir = Path(os.environ.get("VSC_DATA"))  # type: ignore
    scratch_dir = Path(os.environ.get("VSC_SCRATCH"))  # type: ignore
except TypeError:
    data_dir = Path("./data")
    scratch_dir = Path("./scratch")
cache_dir = scratch_dir / "classificationdata"


def load_data():
    datafile = data_dir / "classificationdata/dataset.csv"

    if not datafile.exists():
        dtypes = {
            "Abstract": str,
            "Title": str,
            "year": int,
            "documentType": str,
            "StoreId": str,
            "disc1": str,
            "disc2": str,
        }
        socab_df = pd.read_csv("Datasets/SocAbstracts.csv", dtype=dtypes)
        eric_df = pd.read_csv("Datasets/ERIC.csv", dtype=dtypes)
        econlit_df = pd.read_csv("Datasets/EconLit.csv", dtype=dtypes)

        # Get clean and relabeled dataframes for each set:
        socab_clean = clean_df(socab_df)
        eric_clean = clean_df(eric_df)
        econlit_clean = clean_df(econlit_df)

        df = pd.concat([socab_clean, eric_clean, econlit_clean])
        df = df.drop(columns=["year", "disc1_x", "disc1_counts", "disc2_counts"])

        df["text"] = df.Abstract.str.cat(df.Title, sep=" ")

        # Transform list to semicolon-separated string prior to saving
        df["disc2_x"] = df.disc2_x.apply(lambda x: ";".join(x))
        df.to_csv(datafile, index=False)

    # Read file and transform back to list format
    df = pd.read_csv(datafile)
    df["disc2_x"] = df.disc2_x.str.split(";")

    return df


def true_labels(df):
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(df["disc2_x"])
    labels = list(mlb.classes_)
    return pd.DataFrame(y_true, columns=labels)


def train_test_split(df, df_true, test_size=0.25):
    # iterative_train_test_split is only deterministic if we call this first
    np.random.seed(RANDOM_SEED)
    # iterative_train_test_split expects a matrix, whereas CountVectorizer
    # needs an iterable over strings
    X_train, y_train, X_test, y_test = iterative_train_test_split(
        df.text.to_frame().values, df_true.values, test_size=0.25
    )
    X_train, X_test = X_train[:, 0], X_test[:, 0]

    return X_train, y_train, X_test, y_test


def setup_preprocessing(tokenizer, use_idf, ngram):
    memory = joblib.Memory(cache_dir, verbose=0)
    clf = ExtraTreesClassifier(
        max_depth=10, n_estimators=2000, random_state=RANDOM_SEED, n_jobs=N_CORES
    )

    return Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenizer, ngram_range=ngram)),
            ("tfidf", TfidfTransformer(use_idf=use_idf)),
            ("selectfrommodel", SelectFromModel(clf)),
        ],
        memory=memory,
    )


def fit(X_train, y_train, config, params, n_iter, scoring="accuracy"):
    clf = RandomizedSearchCV(
        ClassifierChain(require_dense=[False, True]),
        params,
        scoring=scoring,
        n_iter=n_iter,
        n_jobs=N_CORES,
	    pre_dispatch=15,
        verbose=2,
        return_train_score=True,
        cv=cv,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)

    # Save CV report
    clf_name = params["classifier"][0].__class__.__name__
    location = cache_dir / f"{clf_name}-{config}-featuresel-cv.csv"
    pd.DataFrame(clf.cv_results_).to_csv(location)

    # Persist classifier to disk
    location = cache_dir / f"{clf_name}-{config}-featuresel.pkl"
    joblib.dump(clf.best_estimator_, location, compress=1)
    return clf


def best_classifier(classifiers):
    best_input, best_clf_loc, best_score = max(
        ((preproc, loc, score) for preproc, (loc, score) in classifiers.items()),
        key=lambda x: x[2],
    )
    return joblib.load(best_clf_loc), best_input, best_score


def make_test_prediction(clf, pipe, X_test):
    pipe.set_params(vect__vocabulary=pipe.named_steps["vect"].vocabulary_)
    X_test_preproc = pipe.transform(X_test)
    return clf.predict(X_test_preproc)


def evaluation_metrics_report(y_test, y_pred):
    return (
        f"Accuracy = {accuracy_score(y_test, y_pred)}\n"
        f"Hamming loss = {hamming_loss(y_test, y_pred)}\n"
        f"Precision = {precision_score(y_test, y_pred, average='weighted')}\n"
        f"Recall = {recall_score(y_test, y_pred, average='weighted')}\n"
        f"F1 = {f1_score(y_test, y_pred, average='weighted')}"
    )


def parse_args():
    tokenizers = {
        "lemmas": tokenize_lemmas,
        "nouns": tokenize_nouns,
        "nounphrases": tokenize_nounphrases,
    }

    parser = argparse.ArgumentParser(description="Classify by gradient boosting")
    parser.add_argument("--tokenizer", choices=tokenizers.keys())
    parser.add_argument("--idf", choices=["1", "0"], default=False)
    parser.add_argument("--bigrams", choices=["1", "0"], default=False)
    args = parser.parse_args()

    tokenizer = tokenizers[args.tokenizer]
    ngram = (1, 2) if args.bigrams == "1" else (1, 1)
    use_idf = True if args.idf == "1" else False

    return tokenizer, use_idf, ngram


if __name__ == "__main__":
    df = load_data()

    df_true = true_labels(df)
    X_train, y_train, X_test, y_test = train_test_split(df, df_true)

    tokenizer, use_idf, ngram = parse_args()
    config = f"{tokenizer.__name__}-{use_idf}-{ngram}"

    pipe = setup_preprocessing(tokenizer, use_idf, ngram)
    # Preprocess training data set
    X_train_preproc = pipe.fit_transform(X_train, y_train)

    ### Multinomial Naive Bayes
    mnb_parameters = {
        "classifier": [MultinomialNB()],
        "classifier__alpha": scipy.stats.reciprocal(1e-4, 1),  # loguniform
    }

    cv = KFold(n_splits=3, random_state=RANDOM_SEED, shuffle=False)
    clf = fit(
        X_train_preproc, y_train, config, mnb_parameters, n_iter=100
    )

    print("\n** CV results for MNB **")
    print(config, clf.best_score_)
    # Make prediction for test set:
    y_pred = make_test_prediction(clf, pipe, X_test)
    y_test = y_test.astype("float32")
    print("\n** Test results for MNB **")
    print(evaluation_metrics_report(y_test, y_pred))

    ### Gradient Boosting
    gb_parameters = {
        "classifier": [
            LGBMClassifier(
                objective="binary",
                boosting_type="gbdt",
                verbose=1,
                boost_from_average=False,
                num_threads=1,
            )
        ],
        "classifier__num_trees": np.arange(100, 800),
        "classifier__num_leaves": np.arange(10, 200),
        "classifier__max_depth": np.arange(5, 25),
        "classifier__learning_rate": scipy.stats.reciprocal(1e-4, 1),
        "classifier__max_bin": np.arange(50, 250),
        "classifier__min_data_in_leaf": np.arange(5, 100),
        "classifier__bagging_fraction": scipy.stats.uniform(0, 1),
        "classifier__bagging_freq": np.arange(10, 50, 10),
        "classifier__min_child_weight": scipy.stats.reciprocal(1e-3, 1e-1),
        "classifier__reg_alpha": scipy.stats.reciprocal(1e-5, 1),
        "classifier__reg_lambda": np.arange(1, 10),
    }

    cv = KFold(n_splits=3, random_state=RANDOM_SEED, shuffle=False)
    clf = fit(
        X_train_preproc, y_train, config, gb_parameters, n_iter=100
    )

    print("\n** CV results for Gradient Boosting **")
    print(config, clf.best_score_)
    # Make prediction for test set:
    y_pred = make_test_prediction(clf, pipe, X_test)
    y_test = y_test.astype("float32")
    print("\n** Test results for Gradient Boosting **")
    print(evaluation_metrics_report(y_test, y_pred))
