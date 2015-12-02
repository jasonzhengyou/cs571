import nltk
from nltk.classify import SklearnClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import cPickle as pickle
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

"""READING IN FROM FILES"""
def read_train(source):
    # open data file
    # discard header line
    fdata = open(source, 'r')
    fdata.readline()
    
    phrases = []
    for data in fdata:
        data = data.split('\t')
        phrases.append(([token.lower().rstrip() for token in data[2].split(' ')],int(data[3].rstrip())))
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
    for i, word in enumerate(global_features):
        features[i] = int(word in words)
        #features['contains(%s)' % word] = (word in words)
    return features

def main():
    print 'reading training data'
    train = pd.read_csv('dat/trainMN.tsv',sep = '\t')
	test = pd.read_csv('dat/devNM.tsv', sep = '\t')
    count_vector = CountVectorizer()
    train_counts = count_vector.fit_transform(train['Phrase'])
    tfidf_vectorizer = TfidfVectorizer(min_df=5,
                             max_df = 0.8,
                             sublinear_tf=True,
                             use_idf=True)
    train_vectors = tfidf_vectorizer.fit_transform(train_counts)
    test_vectors = tfidf_vectorizer.transform(train_vectors)
    classifier = MultinomialNB().fit(x_train_tfidf, train['Sentiment'])
    
    test_counts = count_vector.transform(test['Phrase'])
	test_vectors = tfidf_transformer.transform(test_counts)
	predicted = clf.predict(test_vectors)
    
    for i in predicted:
        print i
    
    pickle.dump(clf, open('multinomialNB.pickle', 'w'))
    
main()