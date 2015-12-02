import nltk
from nltk.classify import SklearnClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import cPickle as pickle
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
import pandas as pd
import numpy as np

all_words = []

def extract_features_dict(text=None, is_raw=False):
    if is_raw:
        text = text.split(' ')

    features = {}
    #word_feature = lambda x: 'contains(%s)' % x

    text_set = set(text)
    #print 'inner feature extraction'
    for i, word in enumerate(all_words):
        if word in text_set:
            features[i] = 1
        else:
            features[i] = 0
    
    #print 'returning dict features'
    return features

def use_feature_dicts(train_x, is_raw = True):
    train_x_dicts = []
    for x in train_x:
        #print 'extracting dict'
        train_x_dict = extract_features_dict(x, is_raw = is_raw)
        train_x_dicts.append(train_x_dict)
    # print train_x_dicts
    vec = DictVectorizer()
    #print 'transforming...'
    return vec.fit_transform(train_x_dicts)

def word_to_set(phrase_list, is_raw=False, trim = 1):
    words_list = []
    words_count = {}
    for phrase in phrase_list:
        for word in phrase.split():
            token = word.lower().rstrip()
            if token in words_count:
                words_count[token] += 1
            else:
                words_count[token] = 1
            if words_count[token] == trim:
                words_list.append(token)

    return list(set(words_list))

def main(clf):
    #print 'getting train'
    train = pd.read_csv('dat/trainMN.tsv',sep = '\t')
    #print 'getting test'
    test = pd.read_csv('dat/devMN.tsv', sep = '\t')

    global all_words
    all_words = word_to_set(train['Phrase'], trim=20, is_raw=True)

    #print 'creating x dict vectors from train'
    train_x = train['Phrase']
    #print 'extracting...'
    train_x = use_feature_dicts(train_x)
    # print train_x

    #print 'creating train y'
    train_y = [int(y) for y in train['Sentiment']]
    if clf == 'NB':
        classifier = MultinomialNB().fit(train_x, train_y)
    elif clf == 'RF':
        classifier = RandomForestClassifier().fit(train_x, train_y)
    elif clf == 'LG':
        classifier = linear_model.LinearRegression()
        classifier = classifier.fit(train_x, train_y)
    elif clf == 'SGD':
        classifier = SGDClassifier().fit(train_x, train_y)
    #print 'testing'
    test_x = use_feature_dicts(test['Phrase'])
    
    for i in classifier.predict(test_x):
        print i
    title = clf + '.pickle'
    pickle.dump(classifier, open(title, 'w'))

main(clf='RF')