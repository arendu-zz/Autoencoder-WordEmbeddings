__author__ = 'arenduchintala'
import gzip
import pdb

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


import utils

if __name__ == '__main__':
    # script here
    funcwords = set(open('functionwords.txt', 'r').read().split('\n'))
    vocab = set([])
    vocab_id = {}
    print 'making vocab list...'
    for e in parse('Arts.demo2.txt.gz'):
        if 'review/text' in e:
            txt = set(e['review/text'].lower().split())
            tokens = txt - funcwords
            vocab.update(tokens)

    for idx, token in enumerate(vocab):
        vocab_id[token] = len(vocab_id)
    print 'made vocab list.'
    print 'reading documents...'
    data = []
    for e in parse('Arts.demo.txt.gz'):
        if 'review/text' in e:
            txt = set(e['review/text'].lower().split())
            tokens = txt - funcwords
            sparse_bit_vector = [vocab_id[t] for t in tokens]
            bt = [0.0] * len(vocab_id)
            for i in sparse_bit_vector:
                bt[i] = 1.0
            bt = np.reshape(bt, (len(bt), 1))
            data.append((bt, bt))

    data = data[:10]
    print len(vocab_id), len(data)
    print 'read documents'
    autoencoder = L.Network(0.1, [len(vocab_id), 50, len(vocab_id)], data)
    init_weights = autoencoder.get_network_weights()
    init_cost = autoencoder.get_cost(init_weights, data)
    print len(init_weights)
    # pdb.set_trace()
    #print 'computing finite difference grad...'
    #grad = autoencoder.get_gradient(init_weights, data)
    #grad_approx = utils.gradient_checking(init_weights, 1e-5, autoencoder.get_cost, data)
    #print 'cosine similarity between grad and finite difference approx', utils.cosine_sim(grad, grad_approx)

    final_weights = autoencoder.train(data)
    # print final_weights
    final_cost = autoencoder.get_cost(final_weights, data)
    print 'cost before training', init_cost, ' after training:', final_cost





