__author__ = 'arenduchintala'
import gzip
import simplejson
import Layers
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


if __name__ == '__main__':
    # script here
    funcwords = set(open('functionwords.txt', 'r').read().split('\n'))
    vocab = set([])
    vocab_id = {}

    c = 0.0
    for e in parse('Arts.txt.gz'):
        if 'review/text' in e:
            txt = set(e['review/text'].lower().split())
            tokens = txt - funcwords
            vocab.update(tokens)
            c += 1
            if c > 50:
                break

    for idx, token in enumerate(vocab):
        vocab_id[token] = len(vocab_id)

    data = []
    for e in parse('Arts.txt.gz'):
        if 'review/text' in e and len(data) < 50:
            txt = set(e['review/text'].lower().split())
            tokens = txt - funcwords
            sparse_bit_vector = [vocab_id[t] for t in tokens]
            bt = [0.0] * len(vocab_id)
            for i in sparse_bit_vector:
                bt[i] = 1.0
            data.append((bt, bt))

    print len(vocab_id), len(data)
    autoencoder = Layers.Network(0.001, [len(vocab_id), 10, len(vocab_id)], data)
    init_weights = autoencoder.get_layer_weights()
    print 'cost', autoencoder.get_cost(init_weights)

    (xopt, fopt, return_status) = fmin_l_bfgs_b(autoencoder.get_cost, init_weights, autoencoder.get_gradient,
                                                pgtol=0.01)
    # print xopt
    print '\nafter training:'
    print autoencoder.get_cost(np.asarray(xopt))
    autoencoder.predict(scale=True)

