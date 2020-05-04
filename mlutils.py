import os

import pandas as pd
import spacy
from nltk.stem import SnowballStemmer
from spacy.lang.en import English

nlp = spacy.load("en_core_web_sm", disable=["ner"])
stop_words = nlp.Defaults.stop_words
stemmer = SnowballStemmer("english")


def is_stop_or_punct(tok):
    """Determine "f token is punctuation or stop word"""
    # Normally we would use tok.is_stop but that has a bug in spaCy < 2.1.0
    # See https://github.com/explosion/spaCy/pull/1891
    return tok.is_punct or tok.lower_ in stop_words


def tokenize_lemmas(abstract):
    """Tokenize into stemmed lemmas"""
    with nlp.disable_pipes("tagger", "parser"):
        doc = nlp(abstract)
    return [stemmer.stem(tok.lower_) for tok in doc if not is_stop_or_punct(tok)]


def tokenize_nouns(abstract):
    """Tokenize into stemmed nouns"""
    with nlp.disable_pipes("parser"):
        doc = nlp(abstract)
    return [
        stemmer.stem(tok.lemma_)
        for tok in doc
        if tok.pos_ == "NOUN" and not is_stop_or_punct(tok)
    ]


def tokenize_nounphrases(abstract):
    """Tokenize into stemmed noun phrases"""
    doc = nlp(abstract)
    nps = []
    for nc in doc.noun_chunks:
        # Dropping stop words from middle of NP may change meaning!
        np_tokens = (
            stemmer.stem(tok.lemma_) for tok in nc if not is_stop_or_punct(tok)
        )
        nps.append(" ".join(np_tokens))
    return nps


def read_dir_into_df(basedir):
    """Reads all files in dir into one dataframe"""
    df = pd.DataFrame()

    for fname in os.listdir(basedir):
        df_tmp = pd.read_excel(os.path.join(basedir, fname))
        df_tmp["disc1"] = fname[:-4]
        df_tmp["disc2"] = fname[:6]
        df = df.append(df_tmp, sort=True)

    return df


def clean_df(dirtydf):
    """takes a dirty set of metadata and returns a clean frame"""
    # 1. Remove incomplete, erroneous or out of scope records:
    # - without abstract or title
    # - with publication date before 2000 and after 2018
    # - with 'very long'  or 'very short' abstracts (<1000 AND > 50)
    dirtydf = dirtydf.dropna(axis=0, subset=["Abstract", "Title"])
    dirtydf = dirtydf[dirtydf.year.between(2000, 2018)]

    dirtydf["n_tokens"] = dirtydf.Abstract.str.count(r"\w+")
    dirtydf = dirtydf[dirtydf.n_tokens.between(50, 1000)]

    # 2. Filter relevant publication types
    # We manually created specific lists of relevant publication types
    relevant_type_lst = {
        line.rstrip("\n") for line in open("relevanttypes_lst.txt", "r")
    }
    dirtydf = dirtydf[dirtydf.documentType.isin(relevant_type_lst)]

    # 3. Aggregate duplicates and re-label abstracts
    # group duplicates and append all labels into one column
    disc1_labels = (
        dirtydf.groupby("StoreId")["disc1"].agg(lambda x: ", ".join(x)).reset_index()
    )
    disc2_labels = (
        dirtydf.groupby("StoreId")["disc2"].agg(lambda x: ", ".join(x)).reset_index()
    )

    # deduplicate level 2 discipline codes
    disc_list = [sorted({x.strip() for x in row.split(",")}) for row in disc2_labels.disc2]
    disc2_labels["disc2"] = disc_list

    # get all labels into one dataframe and count
    discs = disc1_labels.merge(disc2_labels, on="StoreId")
    discs["disc1_counts"] = discs.disc1.str.count(",") + 1
    discs["disc2_counts"] = discs.disc2.apply(len)

    # 4. Merge labels dataframe with 'dirty' df
    cleandf = discs.merge(dirtydf, on="StoreId", how="left").drop_duplicates(
        subset="StoreId"
    )
    return cleandf[
        [
            "Abstract",
            "Title",
            "year",
            "StoreId",
            "disc1_x",
            "disc2_x",
            "disc1_counts",
            "disc2_counts",
        ]
    ]


def get_randomsample(df, random_state=None):
    # 1. get at least two abstracts from each stratum -> stratified sample
    strat_df = df.groupby("disc1_x", group_keys=False).apply(
        lambda x: x.sample(min(len(x), 2))
    )
    # 2. draw random sample from stratified sample.
    sample_df = strat_df.sample(n=50, random_state=random_state).reset_index(drop=True)

    return sample_df[["Abstract", "StoreId", "Title", "disc1_x", "disc2_x"]]
