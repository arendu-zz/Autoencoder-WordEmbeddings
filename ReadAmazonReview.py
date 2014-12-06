__author__ = 'arenduchintala'
import gzip
import pdb
from nltk.tokenize import word_tokenize

try:
    import simplejson
except ImportError:
    import json as simplejson

import NpLayers as L

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


def make_vocab(path_to_corpus, path_to_funcwords):
    funcwords = set(open(path_to_funcwords, 'r').read().split('\n'))
    vocab = set([])
    vocab_id = {}
    print 'making vocab list...'
    for e in parse(path_to_corpus):
        if 'review/text' in e:
            s = e['review/text']
            txt = set([t.lower() for t in word_tokenize(s)])
            tokens = txt - funcwords
            vocab.update(tokens)

    write_vocab_map = open('vocab.map', 'w')
    for idx, token in enumerate(vocab):
        vocab_id[token] = len(vocab_id)
        write_vocab_map.write(token + '\t' + str(vocab_id[token]) + '\n')
    write_vocab_map.flush()
    write_vocab_map.close()
    return vocab_id


def make_data(path_to_corpus, path_to_funcwords):
    funcwords = set(open(path_to_funcwords, 'r').read().split('\n'))
    data = []
    for e in parse(path_to_corpus):
        if 'review/text' in e:
            s = e['review/text']
            txt = set([t.lower() for t in word_tokenize(s)])
            tokens = txt - funcwords
            sparse_bit_vector = [vocab_id[t] for t in tokens]

            bt = [0.0] * len(vocab_id)
            for i in sparse_bit_vector:
                bt[i] = 1.0
            bt = np.reshape(bt, (len(bt), 1))
            data.append((bt, bt))
    return data


import utils

if __name__ == '__main__':
    # script here
    vocab_id = make_vocab('Arts.demo2.txt.gz', 'functionwords.txt')
    print 'reading documents...'
    data = make_data('Arts.demo2.txt.gz', 'functionwords.txt')
    # data = data[:10]
    print len(vocab_id), len(data)
    print 'read documents'
    autoencoder = L.Network(0.1, [len(vocab_id), 50, len(vocab_id)], data)
    init_weights = autoencoder.get_network_weights()
    init_cost = autoencoder.get_cost(init_weights, data)
    print len(init_weights)
    # print 'computing finite difference grad...'
    # grad = autoencoder.get_gradient(init_weights, data)
    # grad_approx = utils.gradient_checking(init_weights, 1e-5, autoencoder.get_cost, data)
    # print 'cosine similarity between grad and finite difference approx', utils.cosine_sim(grad, grad_approx)

    final_weights = autoencoder.train(data, tol=0.01)
    # print final_weights
    final_cost = autoencoder.get_cost(final_weights, data)
    print 'cost before training', init_cost, ' after training:', final_cost





