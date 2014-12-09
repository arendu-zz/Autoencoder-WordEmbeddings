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


def make_vocab(path_to_corpus, path_to_funcwords, max_vocab=5000, save_vocab_map='vocab.map'):
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

    write_vocab_map = open(save_vocab_map, 'w')
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
            data.append((bt, bt))
    return data


import utils

if __name__ == '__main__':
    SAVE_TRAINED_NN = "arts-5000-50.nn"  # give a better name here
    # script here
    dataset = "Arts.demo2.txt.gz"
    print 'making vocab...'
    vocab_id = make_vocab(dataset, 'functionwords.txt', max_vocab=2000)
    print 'reading documents...'
    data = make_data(dataset, vocab_id)

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
    autoencoder.set_network_weights(final_weights)
    L.dump(autoencoder, SAVE_TRAINED_NN)

    final_weights_agagrad = autoencoder.train_adagrad(data, init_weights, 2)
    final_cost_adagrad = autoencoder.get_cost(final_weights_agagrad, data)
    print 'cost before training', init_cost, 'after adagrad training', final_cost_adagrad





