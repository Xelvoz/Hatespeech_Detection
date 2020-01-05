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
import sys
import tweepy


STOPWORDS = set(stopwords.words('english'))
consumer_key = "B5jumiycl6z4OHI9mYEODJYbL"
consumer_secret = "rNvxtfNR9nrDj4HfPP8SFMIK26UmoqiWWlUTlFvyj9djG8vSzI"
access_token = "2893083375-8EBkQDuKCo3spRy1gofGnpcXbebMU3hi9nGQrzV"
access_token_secret = "GGK56xSLdDufVJFSgJqWLmW6Tv7JZ3Y00p0jYkuAfOz9q"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
TWITTER = tweepy.API(auth)


class PreProcessing:

    def __init__(self):
        pass

    @staticmethod
    def sanitize_tweet(tweet, with_stemmer=True, stemmer="porter"):
        # Choix des Stemmer pour avoir la racine d'un mot.
        if stemmer == "snowball":
            STEMMER = SnowballStemmer("english")
        else:
            STEMMER = PorterStemmer()
        text = re.sub("\"+", "", tweet)  # Enlève plusieurs " de suite.
        # Enlève plusieurs espaces de suite et le remplacer par un seul.
        text = re.sub("\s+", " ", text)
        text = re.sub("RT", "", text)  # Enlève RT (ancronyme des Re-Tweets)
        text = re.sub("https?:?/?/?[^\s]+", "", text)  # Enlève les liens
        # Enlève les caractères spéciaux codés en HTML.
        text = re.sub("&.+;", "", text)
        # Initialisation du TweetTokenizer
        # @param preserve_case=False : Tout est en miniscule.
        # @param strip_handles=True : Enlève les mentions.
        # @param reduce_len=True : Réduit une séquence de lettre successive dans un mot en 3 si la longueur de cette séquence est supérieur à 3.
        tknzr = TweetTokenizer(preserve_case=False,
                               strip_handles=True, reduce_len=True)
        # On prend les mots constitués de lettre uniquement tout en enlevant les stopwords.
        text = " ".join([STEMMER.stem(s)
                         if with_stemmer
                         else s
                         for s in tknzr.tokenize(text)
                         if s.isalpha() and s not in STOPWORDS])
        return text

    @staticmethod
    def get_tweet_and_sanitize(id):
        try:
            status = TWITTER.get_status(id)
            return PreProcessing.sanitize_tweet(status.text)
        except:
            return np.nan

    @staticmethod
    def read_davidson_data(path_to_csv):
        data = pd.read_csv(path_to_csv)
        data = pd.concat([data["class"], data["tweet"]], axis=1)
        data.replace(
            {0: "hate speech", 1: "offensive language", 2: "normal"}, inplace=True)
        return data

    @staticmethod
    def read_waseem_hovy_data(path_to_csv):
        new = pd.read_csv(path_to_csv)
        return new

    @staticmethod
    def read_waseem_hovy_data(path_to_csv):
        data = pd.read_csv(path_to_csv)
        data = pd.concat([data.label, data.tweet_id], axis=1)
        data.columns = ["class", "tweet"]
        return data.dropna()


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

    def check_ngrams(self, max_ngrams=10):
        classes = self.data["class"].unique()
        old_data = self.data.copy()

        stats = pd.DataFrame(
            columns=["ngrams", "class", "precision", "f1-score", "recall"])
        for i in range(1, max_ngrams+1):
            data = old_data.copy()
            clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, i))), ('tfidf', TfidfTransformer(
            )), ('clf', SGDClassifier(class_weight="balanced", shuffle=True)), ])
            clf = clf.fit(data["tweet"], data["class"])
            report = metrics.classification_report(
                old_data["class"], clf.predict(old_data["tweet"]), output_dict=True)
            for c in classes:
                L = report[c]
                stats = stats.append(
                    {"ngrams": i, "class": c, "precision": L["precision"], "f1-score": L["f1-score"], "recall": L["recall"]}, ignore_index=True)
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
