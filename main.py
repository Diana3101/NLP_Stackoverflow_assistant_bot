import sys

sys.path.append("..")
from common.download_utils import download_project_resources

download_project_resources()
from utils import *

import numpy as np
import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


def tfidf_features(X_train, X_test, vectorizer_path):
    """Performs TF-IDF transformation and dumps the model."""

    # Train a vectorizer on X_train data.
    # Transform X_train and X_test data.

    # Pickle the trained vectorizer to 'vectorizer_path'
    # Don't forget to open the file in writing bytes mode.

    tfidf_vectorizer = TfidfVectorizer(token_pattern='(\\S+)', min_df=5, max_df=0.9, ngram_range=(1, 2))
    fit_tfidf = tfidf_vectorizer.fit(X_train)
    X_train = fit_tfidf.transform(X_train)
    X_test = fit_tfidf.transform(X_test)
    pickle.dump(fit_tfidf, open(vectorizer_path, 'wb'))

    return X_train, X_test


sample_size = 200000

dialogue_df = pd.read_csv('data/dialogues.tsv', sep='\t').sample(sample_size, random_state=0)
stackoverflow_df = pd.read_csv('data/tagged_posts.tsv', sep='\t').sample(sample_size, random_state=0)

dialogue_df.head()
stackoverflow_df.head()

from utils import text_prepare

dialogue_df['text'] = dialogue_df.text.map(text_prepare)
stackoverflow_df['title'] = stackoverflow_df.title.map(text_prepare)

from sklearn.model_selection import train_test_split

X = np.concatenate([dialogue_df['text'].values, stackoverflow_df['title'].values])
y = ['dialogue'] * dialogue_df.shape[0] + ['stackoverflow'] * stackoverflow_df.shape[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

X_train_tfidf, X_test_tfidf = tfidf_features(X_train, X_test, RESOURCE_PATH['TFIDF_VECTORIZER'])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

intent_recognizer = LogisticRegression(penalty='l2', C=10, random_state=0).fit(X_train_tfidf, y_train)

# Check test accuracy.
y_test_pred = intent_recognizer.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy))

pickle.dump(intent_recognizer, open(RESOURCE_PATH['INTENT_RECOGNIZER'], 'wb'))

X = stackoverflow_df['title'].values
y = stackoverflow_df['tag'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

vectorizer = pickle.load(open(RESOURCE_PATH['TFIDF_VECTORIZER'], 'rb'))

X_train_tfidf, X_test_tfidf = vectorizer.transform(X_train), vectorizer.transform(X_test)

from sklearn.multiclass import OneVsRestClassifier

tag_classifier = OneVsRestClassifier(LogisticRegression(penalty='l2', C=5, random_state=0)).fit(X_train_tfidf,
                                                                                                y_train)

# Check test accuracy.
y_test_pred = tag_classifier.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy))

pickle.dump(tag_classifier, open(RESOURCE_PATH['TAG_CLASSIFIER'], 'wb'))

starspace_embeddings, embeddings_dim = load_embeddings('data/word_embeddings.tsv')

posts_df = pd.read_csv('data/tagged_posts.tsv', sep='\t')

counts_by_tag = posts_df[['tag', 'post_id']].groupby(['tag', ]).count().to_dict()['post_id']

import os

os.makedirs(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], exist_ok=True)

for tag, count in counts_by_tag.items():
    tag_posts = posts_df[posts_df['tag'] == tag]

    tag_post_ids = tag_posts.post_id.tolist()

    tag_vectors = np.zeros((count, embeddings_dim), dtype=np.float32)
    for i, title in enumerate(tag_posts['title']):
        tag_vectors[i, :] = question_to_vec(title, starspace_embeddings, embeddings_dim)

    # Dump post ids and vectors to a file.
    filename = os.path.join(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], os.path.normpath('%s.pkl' % tag))
    pickle.dump((tag_post_ids, tag_vectors), open(filename, 'wb'))
