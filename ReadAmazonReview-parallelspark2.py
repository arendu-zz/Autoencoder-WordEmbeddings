__author__ = 'arenduchintala'
import gzip, sys, itertools
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

def train(itr, bae):
    ae = bae.value
    xs = []
    for x in itr:
      xs.append(x)
    (xopt, fopt, return_status) = fmin_l_bfgs_b(ae.get_cost, 
                                                ae.get_network_weights(), 
                                                ae.get_gradient, args=(xs, ),
                                                pgtol=0.1, maxfun=5)
    print "xopt=", xopt
    print "cost in train=", ae.get_cost(xopt, xs)
    return 1, xopt

# This isn't being used right now, and is probably useless
def train_alt(itr, bae):
    ae = bae.value
    itrs = itertools.tee(itr, 4)
    for itr in itrs:
      xs = []
      for x in itr:
        xs.append(x)
      (xopt, fopt, return_status) = fmin_l_bfgs_b(ae.get_cost, 
                                                ae.get_network_weights(), 
                                                ae.get_gradient, args=(xs, ),
                                                pgtol=0.1, maxfun=5)
      print "xopt=", xopt
      print "cost in train=", ae.get_cost(xopt, xs)
      yield 1, xopt

# This isn't being used right now
def merge(x, y):
  print "x =", x, " y=", y
  x_w = x[1]
  y_w = y[1]
  return np.add(x_w, y_w)

if __name__ == '__main__':
    #input_width = 2000
    inside_width = 100

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

    input_width = len(vocab_id)
    #input_width = 2000
   
    print "input width =", input_width
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

    # supposed to be this
    #data = data[:5000]
    data = data[:500]

    num_chunks = 4
    threshold = 10
    converged = False
    # This is the model we care about. The weights of this model will be updated to reflect the
    # average of multiple autoencoders (with identical topology) which are trained on different
    # subsets of the the training data.
    ae = L.Network(0.1, [input_width, inside_width, input_width])
    print "Size of ae =", ae.size_bytes()
    prev_cost = ae.get_cost(ae.get_network_weights(), data)
    itr = 0
    # Removed:
    #      .setMaster("local")
    conf = (SparkConf()
         .setAppName("My app"))
        # .set("spark.executor.memory", "10g")
        # .set("spark.python.worker.memory","10g"))
    sc = SparkContext(conf=conf)

    rdd = sc.parallelize(data, numSlices=int(num_chunks))
    weights = None
    while itr < 1:
        cost = 0.0
        bae = sc.broadcast(ae)
        weights = rdd.mapPartitions(lambda x: train(x, bae)).collect()
        print "weights = ", weights
        summed_weights = weights[1]
        for i in range(2, len(weights)):
          if i % 2 == 0:
            continue
          print "weights[i]=", weights[i]
          summed_weights = np.add(summed_weights, weights[i])
          print "weights after summing=", summed_weights
        new_weights = summed_weights / num_chunks
        print "new_weights =", new_weights
        ae.set_network_weights(new_weights)
        cost = ae.get_cost(new_weights, data)
        print "sys size ae =", ae.size_bytes()

        print "Cost:", cost
        print "Prev Cost:", prev_cost
        # At the moment it is better to see how the cost is changing with repeated iterations,
        # rather than stopping early. 
        if abs(cost - prev_cost) < threshold:
            #converged = True
            print "Training has converged"
        prev_cost = cost
        itr += 1
