# CS 600.615 Big Data
#
# This script demonstrates parallel autoencoder training in Pyspark
# using a simple averaging method.
# See NpLayers.py for the autoencoder implementation.
#
# Authors: David Snyder, Adithya Renduchintala, Rebecca Knowles

import gzip, sys, itertools, time
import pdb
import nltk
from nltk.tokenize import word_tokenize
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process
import numpy as np
import time

try:
    import simplejson
except ImportError:
    import json as simplejson

import NpLayers as L
from scipy.optimize import fmin_l_bfgs_b
import numpy as np

global autoencoder, accum, cpu_count
cpu_count = None
accum = None
autoencoder = None

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

def para_decode(idx,chunk):
    global autoencoder
    print "Decoding", idx

    return (idx, [autoencoder.get_representation(d) for d in chunk])
def para_accum(pair):
    global accum
    print "Got from index: ",pair[0]
    accum[pair[0]]=pair[1]

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
    ae = sys.argv[2]
    wordmap = sys.argv[3]
    increase_data = sys.argv[4]
    cpu_count = int(sys.argv[5])
    
    accum = range(cpu_count)
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
    data = make_data(decoding_data, vocab_id)[:1000]
    full_data = data*int(increase_data)
    print 'done reading documents', len(full_data), 'documents...'
    start_time = time.time()

    num_chunks=cpu_count
    data = []
    for c in xrange(int(num_chunks)):
        data_chunk = full_data[c * int(len(full_data) / num_chunks): (c + 1) * int(len(full_data) / num_chunks)]
        data.append(data_chunk)

    pool = Pool(processes=cpu_count)
    print "preparing to decode"
    for idx, chunk in enumerate(data):
        pool.apply_async(para_decode, args=(idx,chunk), callback = para_accum)
    pool.close()
    pool.join()
    print "Finished."
    elapsed_time = time.time() - start_time
    print "Time:",elapsed_time

