# CS 600.615 Big Data
#
# This script demonstrates parallel autoencoder decoding in Pyspark.
# See Autoencoder.py for the autoencoder implementation.
#
# Authors: Rebecca Knowles, David Snyder, Adithya Renduchintala

from pyspark import SparkContext, SparkConf
from TrainSerial import parse
import gzip

try:
    import simplejson
except ImportError:
    import json as simplejson

import Autoencoder as L
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
if __name__ == '__main__':
    #Same as previous to train the model
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

    short_data = data[:200]

    print len(vocab_id), len(short_data)
    print 'read documents'
    autoencoder = L.Network(0.1, [len(vocab_id), 5, len(vocab_id)], short_data)
    init_weights = autoencoder.get_layer_weights()
    init_cost = autoencoder.get_cost(init_weights)

    (xopt, fopt, return_status) = fmin_l_bfgs_b(autoencoder.get_cost, init_weights, autoencoder.get_gradient, pgtol=0.1,
                                                maxfun=10)
    # print xopt
    final_cost = autoencoder.get_cost(np.asarray(xopt))
    print 'cost before training', init_cost, ' after training:', final_cost
    """
    #Serial implementation
    for i in range(10):
        for item in data[:1000]:
            print autoencoder.get_representation(item[0])
    """
    #Spark parallel implementation
    conf = SparkConf().setAppName("test").setMaster("local")
    sc = SparkContext(conf=conf)
    distData = sc.parallelize([item[0] for item in data][:1000],25)
    res = distData.map(lambda s: (s,autoencoder.get_representation(s)))
    output = res.collect()
    print len(output)
    #for item in output:
    #    print item[1]

