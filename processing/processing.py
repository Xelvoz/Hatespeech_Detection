from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn import metrics
from nltk.stem.porter import PorterStemmer
import sklearn as skl
import numpy as np
import pandas as pd
import seaborn as sns
import re
import os

STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()


class PreProcessing:

    def __init__(self):
        pass

    @staticmethod
    def sanitize_tweet(tweet):
        text = re.sub("\"+", "", tweet)
        text = re.sub("\s+", " ", text)
        text = re.sub("RT", "", text)
        text = re.sub("https?:?/?/?[^\s]+", "", text)
        text = re.sub("&#[0-9]+;", "", text)
        text = re.sub("&.+;", "", text)
        tknzr = TweetTokenizer(preserve_case=False, strip_handles=True)
        text = " ".join([STEMMER.stem(s) for s in tknzr.tokenize(
            text) if s.isalpha() and s not in STOPWORDS])
        return text

    @staticmethod
    def read_davidson_data(path_to_csv):
        data = pd.read_csv(path_to_csv)
        data = pd.concat([data["class"], data["tweet"]], axis=1)
        data.replace(
            {0: "hate speech", 1: "offensive language", 2: "normal"}, inplace=True)
        return data

    @staticmethod
    def read_stormfrontws_data(path_to_csv, path_to_files_dir):

        def read_text(file_id):
            location = os.path.join(path_to_files_dir, file_id+".txt")
            if (os.path.exists(location)):
                with open(location, "r") as f:
                    return PreProcessing.sanitize_tweet(f.read().strip())
            return ""

        data = pd.read_csv(path_to_csv)
        new = pd.DataFrame()
        new["class"] = data["label"].replace(
            {"noHate": "normal", "hate": "hate speech"})
        new["tweet"] = data["file_id"].apply(read_text)
        return new
