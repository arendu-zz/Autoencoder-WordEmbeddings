__author__ = 'arenduchintala'
import gzip
from pyspark import SparkContext, SparkConf

try:
    import simplejson
except ImportError:
    import json as simplejson

import NpLayers as L
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

def train(ae, weights):
    #Set layer weights first (from average AE)
    if weights!= None:
        ae.set_layer_weights(weights)
    (xopt, fopt, return_status) = fmin_l_bfgs_b(ae.get_cost, 
                                                ae.get_layer_weights(), 
                                                ae.get_gradient,
                                                pgtol=0.1, maxfun=5)
    ae.set_layer_weights(xopt)
    #cost += ae.get_cost(xopt)
    #print "idx =", idx, "cost =", ae.get_cost(xopt)
    return ae


def average_weights(aes, num_chunks):
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
    input_width = 2000
    inside_width = 50

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
            bt = bt[:input_width]
            bt = np.reshape(bt, (input_width, 1))
            data.append((bt, bt))

    data = data[:5000]
    print len(vocab_id), len(data)

    num_chunks = 9.0
    autoencoders = []
    for c in xrange(int(num_chunks)):
        data_chunk = data[c * int(len(data) / num_chunks): (c + 1) * int(len(data) / num_chunks)]
        print 'data chunk size', len(data_chunk)
        ae = L.Network(0.1, [input_width, inside_width, input_width], data_chunk)
        autoencoders.append(ae)
    print 'initialized parallel encoders...'

    threshold = 10
    converged = False
    # This is the model we care about. The weights of this model will be updated to reflect the
    # average of multiple autoencoders (with identical topology) which are trained on different
    # subsets of the the training data.
    avg_ae = L.Network(0.1, [input_width, inside_width, input_width], data)
    prev_cost = avg_ae.get_cost(avg_ae.get_layer_weights())
    itr = 0
    conf = (SparkConf()
         .setMaster("local")
         .setAppName("My app")
         .set("spark.executor.memory", "10g")
         .set("spark.python.worker.memory","10g"))
    sc = SparkContext(conf=conf)

    #SPARK: Distribute "data" (actually aes, which include data chunks)
    distData = sc.parallelize(autoencoders,int(num_chunks))

    weights = None
    while itr < 5:
        cost = 0.0
        
        #SPARK: Train models in parallel, updates weights first
        models = distData.map(lambda x: train(x, weights))
        #SPARK: Collect all models -- likely better to use accumulator
        mods = models.collect()
        weights = average_weights(mods, num_chunks)
        avg_ae.set_layer_weights(weights)
        cost = avg_ae.get_cost(weights)

        print "Cost:", cost
        print "Prev Cost:", prev_cost
        # At the moment it is better to see how the cost is changing with repeated iterations,
        # rather than stopping early. 
        if abs(cost - prev_cost) < threshold:
            #converged = True
            print "Training has converged"
        prev_cost = cost
        itr += 1
