import nltk
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
import cPickle as pickle
import time
import pandas as pd

"""READING IN FROM FILES"""
def read_train(source):
    # open data file
    # discard header line
    fdata = open(source, 'r')
    fdata.readline()
    
    phrases = []
    for data in fdata:
        data = data.split('\t')
        phrases.append(([token.lower().rstrip() for token in data[2].split(' ')],int(data[3]))
    return phrases


def read_data(source='', which='train'):

    phrases = {
        'train': read_train,
        'dev': 'dev',   # unimplemented TODO 
        'test': 'test' # unimplemented TODO 
    }[which](source)

    return phrases
    fdata.close()

    
"""WORD AND FEATURE EXTRACTION""" 
def get_phrase_list(sent_tups, with_label=False):
    phrases = []
    for s_tup in sent_tups:
        if not with_label:
            phrases.append(' '.join(s_tup[0]))
        else:
            phrases.append((' '.join(s_tup[0]), s_tup[1]))
    return phrases

def get_all_words(sent_tups):
    all_words = []
    for sent_tup in sent_tups:
        words, _ = sent_tup
        all_words.extend(words)
    return all_words

def get_features(words):
    words = nltk.FreqDist(words)
    features = words.keys()
    return features

global_features = {}

def extract_features(doc):
    words = set(doc)
    features = {}
    for word in global_features:
        features['contains(%s)' % word] = (word in words)
    return features

def read_dev(source):
    df = pd.read_csv(source, sep = '\t')
    dev_sentences = []
    for line in df['Phrases']:
        dev_sentences.append(line)
    return dev_sentences

def main():
    training_data = read_data(source='dat/trainCut.tsv')
    global global_features 
    global_features = get_features(get_all_words(training_data))
    
    training_set = nltk.classify.util.apply_features(extract_features, get_phrase_list(training_data, True)) 
    print 'moving to classifier creation'
    start = time.clock()
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print 'classfier total time: ' + str(time.clock() - start)
    #classifier = SklearnClassifier(BernoulliNB()).train(training_set)
    pickle.dump(classifier, open('classifier.pickle', 'w'))
    for sentence in read_dev('dat/dev.tsv'):
        print classifier.classify(extract_features(sentence.split()))
        
def accuracy():
    correct = 0;
    total = 0;
    for i in numSentencesTested:
        total++
        if test[i] == gold[i]:
            correct++
    print 'accuracy' + correct/total

main()