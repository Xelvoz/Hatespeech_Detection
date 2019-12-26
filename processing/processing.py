from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.base import clone
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import sklearn as skl
import numpy as np
import pandas as pd
import seaborn as sns
import re
import os

STOPWORDS = set(stopwords.words('english'))


class PreProcessing:

    def __init__(self):
        pass

    @staticmethod
    def sanitize_tweet(tweet, with_stemmer=True, stemmer="porter"):
        if stemmer == "snowball":
            STEMMER = SnowballStemmer("english")
        else:
            STEMMER = PorterStemmer()
        text = re.sub("\"+", "", tweet)
        text = re.sub("\s+", " ", text)
        text = re.sub("RT", "", text)
        text = re.sub("https?:?/?/?[^\s]+", "", text)
        text = re.sub("&.+;", "", text)
        tknzr = TweetTokenizer(preserve_case=False,
                               strip_handles=True, reduce_len=True)
        text = " ".join([STEMMER.stem(s) if with_stemmer else s for s in tknzr.tokenize(
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
    def read_stormfrontws_data(path_to_csv):
        data = pd.read_csv(path_to_csv)
        new = pd.DataFrame()
        new["class"] = data[(data["class"] == "hate speech") | (
            data["class"] == "normal")]["class"]
        new["tweet"] = data["tweet"]
        return new


class Analysis:

    def __init__(self, data):
        self.data = data

    def check_metrics_with_less_data(self, cls):
        classes = self.data["class"].unique()
        assert (cls in set(classes))
        old_data = self.data.copy()

        rang = np.arange(1000, old_data.shape[0], 2000)
        stats = pd.DataFrame(
            columns=["inputs", "class", "precision", "f1-score", "recall"])
        for i in rang:
            data = old_data.copy()
            ol = data[data["class"] == cls][:i]
            data = data[data["class"] != cls].append(ol)
            clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))),
                            ('tfidf', TfidfTransformer()), ('clf', SGDClassifier()), ])
            clf = clf.fit(data["tweet"], data["class"])
            report = metrics.classification_report(
                old_data["class"], clf.predict(old_data["tweet"]), output_dict=True)
            for c in classes:
                L = report[c]
                stats = stats.append(
                    {"inputs": i, "class": c, "precision": L["precision"], "f1-score": L["f1-score"], "recall": L["recall"]}, ignore_index=True)
        return stats

    def frequent_words_by_class(self, cls):
        assert (cls in set(self.data["class"].unique()))
        off = CountVectorizer()
        bow = off.fit_transform(self.data.tweet[self.data["class"] == cls])
        sum_words = bow.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in off.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return pd.DataFrame(words_freq, columns=["word", "count"])

    def plot_frequent_words_by_class(self, cls, topn=10):
        data = self.frequent_words_by_class(cls)
        g = sns.barplot(data=data.loc[:topn],
                        x="word", y="count", palette="Set2")
        g.set_title(
            "Word Frequencies for Tweets Labeled as \"{}\". (Top {})".format(cls, topn))
        plt.show()
