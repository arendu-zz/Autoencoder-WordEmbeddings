# CS 600.615 Big Data
#
# This script demonstrates parallel autoencoder training in Pyspark
# using a simple averaging method.
# See NpLayers.py for the autoencoder implementation.
#
# Authors: David Snyder, Adithya Renduchintala, Rebecca Knowles

import gzip, sys, itertools, time
import pdb
#from pyspark import SparkContext, SparkConf
#import nltk
#from nltk.tokenize import word_tokenize
#nltk.download('all')

try:
    import simplejson
except ImportError:
    import json as simplejson

import NpLayers as L
from scipy.optimize import fmin_l_bfgs_b
import numpy as np

def word_tokenize(s):
    return s.split()
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


def make_vocab(path_to_corpus, path_to_funcwords, max_vocab=5000):
    funcwords = set(open(path_to_funcwords, 'r').read().split('\n'))
    vocab_id = {}
    vocab_count = {}
    print 'making vocab list...'
    for e in parse(path_to_corpus):
        if 'review/text' in e:
            s = e['review/text']
            txt = set([t.lower() for t in word_tokenize(s)])
            tokens = txt - funcwords

            for t in tokens:
                vocab_count[t] = vocab_count.get(t, 0.0) + 1.0
    vocab_count_inv = sorted([(c, t) for t, c in vocab_count.items()], reverse=True)[:max_vocab]
    capped_vocab = [t for c, t in vocab_count_inv]

    write_vocab_map = open('vocab.map', 'w')
    for idx, token in enumerate(capped_vocab):
        vocab_id[token] = len(vocab_id)
        write_vocab_map.write(token + '\t' + str(vocab_id[token]) + '\n')
    write_vocab_map.flush()
    write_vocab_map.close()
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
            data.append( bt)
    return data

if __name__ == '__main__':
    decoding_data = sys.argv[1]
    data_out = sys.argv[2]
    ae = sys.argv[3]
    wordmap = sys.argv[4]

    word_out = sys.argv[5]
    # Size of the embeddings.
    inside_width = 50

    print 'making vocab...'
    vocab_id = make_vocab(decoding_data, 'functionwords.txt', max_vocab=2000)
    print 'reading documents...'
    data = make_data(decoding_data,  vocab_id)
    print 'done reading documents', len(data), 'documents...'

    input_width = len(vocab_id)
    print input_width, inside_width

    autoencoder = L.load(ae) #L.Network(0.01, [input_width, inside_width, input_width])
    
    #Decode documents
    decoded = [autoencoder.get_representation(d) for d in data]
    
    f = open(data_out,'w')
    for i in range(len(decoded)):
        f.write(str(i)+'\t'+str(np.reshape(decoded[i],inside_width))+'\n')
    f.close()

    #Decode words
    f = open(word_out,'w')
    for line in open(wordmap).readlines():
        word, val = line.strip().split()
        onehot = [0.0] * len(vocab_id)
        onehot[int(val)] = 1
        onehot = np.reshape(onehot, (len(onehot), 1))
        f.write(word+'\t'+str(np.reshape(autoencoder.get_representation(onehot),inside_width))+'\n')
    f.close()
