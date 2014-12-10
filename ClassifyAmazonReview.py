# CS 600.615 Big Data
#
# This script demonstrates classification using learned representations.
#
# See NpLayers.py for the autoencoder implementation.
#
# Authors: David Snyder, Adithya Renduchintala, Rebecca Knowles

import gzip, sys, itertools, time
import pdb
import nltk
from nltk.tokenize import word_tokenize


try:
    import simplejson
except ImportError:
    import json as simplejson

import NpLayers as L
from scipy.optimize import fmin_l_bfgs_b
import numpy as np

def parse(filename):
    f = gzip.open(filename, 'r')
    entry = {}
    for l in f:
        l = l.strip()
        colonPos = l.find(':')
        if colonPos == -1:
            yield entry
            entry = {}
            continue
        attribute_name = l[:colonPos]
        attribute_val = l[colonPos + 2:]
        entry[attribute_name] = attribute_val
    yield entry

def read_vocab(path_to_vocab):
    vocab_id = {}
    for line in open(path_to_vocab).readlines():
        word,num = line.strip().split()
        vocab_id[word]=int(num)
    return vocab_id

#Produces class-balanced data for classification
def make_data(path_to_corpus, vocab_id):
    data = []
    labels = []
    for e in parse(path_to_corpus):
        if 'review/text' in e:
            s = e['review/text']
            tokens = set([t.lower() for t in word_tokenize(s)])
            sparse_bit_vector = [vocab_id[t] for t in tokens if t in vocab_id]
            bt = [0.0] * len(vocab_id)
            for i in sparse_bit_vector:
                bt[i] = 1.0
            bt = np.reshape(bt, (len(bt), 1))
            data.append( bt)
        if 'review/score' in e:
            s = float(e['review/score'])
            labels.append(s)
    neg_data = []
    neg_labels = []
    pos_data = []
    pos_labels = []
    for i in range(len(data)):
        if labels[i]==1.0:
            neg_labels.append(0)
            neg_data.append(data[i])
        elif labels[i]==5.0:
            pos_labels.append(1)
            pos_data.append(data[i])
    minimum = min(len(neg_data),len(pos_data))
    data = neg_data[:minimum]+pos_data[:minimum]
    labels = neg_labels[:minimum]+pos_labels[:minimum]
    return data, labels

if __name__ == '__main__':
    #Pass as parameters:
    #[path to data to decode] [path to autoencoder] [path to vocab.map] [folds]

    decoding_data = sys.argv[1]
    ae = sys.argv[2]
    wordmap = sys.argv[3]
    folds = int(sys.argv[4])

    #Load trained autoencoder
    autoencoder = L.load(ae)

    print 'making vocab...'
    #vocab_id = make_vocab(decoding_data, 'functionwords.txt', max_vocab=2000)
    vocab_id = read_vocab(wordmap)
    input_width = len(vocab_id)

    # Size of the embeddings.
    inside_width = autoencoder.topology[1]

    assert(input_width == autoencoder.topology[0])

    print 'reading documents...'
    data,labels = make_data(decoding_data,  vocab_id)
    print 'done reading documents', len(data), 'documents...'

    #Decode documents
    decoded = np.array([list(autoencoder.get_representation(d)) for d in data]).reshape((len(data), inside_width))
    data = np.array([list(d) for d in data]).reshape((len(data),input_width))
    labels = np.array(labels)

    from sklearn.cross_validation import cross_val_score
    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB

    #Choose your classifier from these, or others
    clf = MultinomialNB()#SVC(kernel='linear')

    #Score with cross-fold validation
    dec_scores = cross_val_score(clf, decoded, labels, cv=folds)
    print("Representation Accuracy: %0.2f (+/- %0.2f)" % (dec_scores.mean(), dec_scores.std() * 2))

    data_scores = cross_val_score(clf, data, labels, cv=folds)
    print("Raw Data Vect. Accuracy: %0.2f (+/- %0.2f)" % (data_scores.mean(), data_scores.std() * 2))
