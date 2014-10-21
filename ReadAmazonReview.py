__author__ = 'arenduchintala'
import gzip
import simplejson
import NpLayers as L
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import fmin_bfgs
import numpy as np
import pdb


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


if __name__ == '__main__':
    # script here
    funcwords = set(open('functionwords.txt', 'r').read().split('\n'))
    vocab = set([])
    vocab_id = {}

    for e in parse('Arts.demo.txt.gz'):
        if 'review/text' in e:
            txt = set(e['review/text'].lower().split())
            tokens = txt - funcwords
            vocab.update(tokens)

    for idx, token in enumerate(vocab):
        vocab_id[token] = len(vocab_id)

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

    print len(vocab_id), len(data)
    autoencoder = L.Network(0.001, [len(vocab_id), 5, len(vocab_id)], data)
    init_weights = autoencoder.get_layer_weights()
    print 'cost', autoencoder.get_cost(init_weights)

    (xopt, fopt, return_status) = fmin_l_bfgs_b(autoencoder.get_cost, init_weights, autoencoder.get_gradient, pgtol=0.1)
    # print xopt
    print '\nafter training:'
    print autoencoder.get_cost(np.asarray(xopt))
    autoencoder.predict(scale=True)
    # W = autoencoder.layers[0]
    # pdb.set_trace()

