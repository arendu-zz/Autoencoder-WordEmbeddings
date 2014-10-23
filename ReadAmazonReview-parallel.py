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


def average_weights(aes):
    ave_weights = None
    for ae in aes:
        ae_weights = ae.get_layer_weights()
        if ave_weights is None:
            ave_weights = ae_weights
        else:
            ave_weights += ae_weights
    ave_weights *= (1.0 / num_chunks)
    return ave_weights


if __name__ == '__main__':
    # script here
    funcwords = set(open('functionwords.txt', 'r').read().split('\n'))
    vocab = set([])
    vocab_id = {}
    print 'making vocab list...'
    for e in parse('Arts.txt.gz'):
        if 'review/text' in e:
            txt = set(e['review/text'].lower().split())
            tokens = txt - funcwords
            vocab.update(tokens)

    for idx, token in enumerate(vocab):
        vocab_id[token] = len(vocab_id)
    print 'made vocab list.'
    print 'reading documents...'
    data = []
    for e in parse('Arts.demo2.txt.gz'):
        if 'review/text' in e:
            txt = set(e['review/text'].lower().split())
            tokens = txt - funcwords
            sparse_bit_vector = [vocab_id[t] for t in tokens]
            bt = [0.0] * len(vocab_id)
            for i in sparse_bit_vector:
                bt[i] = 1.0
            bt = np.reshape(bt, (len(bt), 1))
            data.append((bt, bt))

    data = data[:200]
    print len(vocab_id), len(data)
    # simulated parallel
    num_chunks = 4.0
    autoencoders = []
    for c in xrange(int(num_chunks)):
        data_chunk = data[c * int(len(data) / num_chunks): (c + 1) * int(len(data) / num_chunks)]
        print 'data chunk size', len(data_chunk)
        ae = L.Network(0.1, [len(vocab_id), 50, len(vocab_id)], data_chunk)
        autoencoders.append(ae)
    print 'initialized parallel encoders...'
    perv_cost = float('inf')
    threshold = 10
    converged = False
    while not converged:
        cost = 0.0
        weights = average_weights(autoencoders)
        for idx, ae in enumerate(autoencoders):
            (xopt, fopt, return_status) = fmin_l_bfgs_b(ae.get_cost, weights, ae.get_gradient,
                                                        pgtol=0.1)
            ae.set_layer_weights(xopt)
            cost += ae.get_cost(xopt)
        print 'cost:', cost
        if abs(cost - perv_cost) < threshold:
            converged = True
        else:
            converged = False
            prev_cost = cost