# CS 600.615 Big Data
#
# This script demonstrates serial autoencoder decoding.
# See Autoencoder.py for the autoencoder implementation.
#
# Authors: Rebecca Knowles, David Snyder, Adithya Renduchintala

import gzip, sys, itertools, time
import pdb
import nltk
from nltk.tokenize import word_tokenize

nltk.download('all')

try:
    import simplejson
except ImportError:
    import json as simplejson

import Autoencoder as L
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
        word, num = line.strip().split()
        vocab_id[word] = int(num)
    return vocab_id


def make_data(path_to_corpus, vocab_id):
    data = []
    for e in parse(path_to_corpus):
        if 'review/text' in e:
            s = e['review/text']
            tokens = set([t.lower() for t in word_tokenize(s)])
            sparse_bit_vector = [vocab_id[t] for t in tokens if t in vocab_id]
            bt = [0.0] * len(vocab_id)
            for i in sparse_bit_vector:
                bt[i] = 1.0
            bt = np.reshape(bt, (len(bt), 1))
            data.append(bt)
    return data


if __name__ == '__main__':
    # Pass as parameters:
    # [path to data to decode] [file to write data embeddings]
    # [path to autoencoder] [path to vocab.map]
    # [file to write word embeddings]

    #Output written to data_out and word_out:
    #    tab-separated: [ID]    [embedding (as list)]
    #    for docs, ID is just index of doc as read in
    #    for words, ID is the word itself
    decoding_data = sys.argv[1]
    data_out = sys.argv[2]
    ae = sys.argv[3]
    wordmap = sys.argv[4]
    word_out = sys.argv[5]

    #Load trained autoencoder
    autoencoder = L.load(ae)

    print 'making vocab...'
    #vocab_id = make_vocab(decoding_data, 'functionwords.txt', max_vocab=2000)
    vocab_id = read_vocab(wordmap)
    input_width = len(vocab_id)

    # Size of the embeddings.
    inside_width = autoencoder.topology[1]

    assert (input_width == autoencoder.topology[0])

    print 'reading documents...'
    data = make_data(decoding_data, vocab_id)
    print 'done reading documents', len(data), 'documents...'

    #Decode documents
    decoded = [autoencoder.get_representation(d) for d in data]

    f = open(data_out, 'w')
    for i in range(len(decoded)):
        arr_str = ','.join([str(n) for n in list(np.reshape(decoded[i], inside_width))])
        f.write(str(i) + '\t'
                + arr_str + '\n')
    f.close()

    #Decode words
    f = open(word_out, 'w')
    for line in open(wordmap).readlines():
        word, val = line.strip().split()
        onehot = [0.0] * len(vocab_id)
        onehot[int(val)] = 1
        onehot = np.reshape(onehot, (len(onehot), 1))
        arr_str = ','.join([str(n) for n in list(np.reshape(autoencoder.get_representation(onehot)
                                                            , inside_width))])
        f.write(word + '\t'
                + arr_str + '\n')
    f.close()
