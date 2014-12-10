# CS 600.615 Big Data
#
# This script demonstrates parallel autoencoder training in Pyspark
# using a simple averaging method.
# See NpLayers.py for the autoencoder implementation.
#
# Authors: David Snyder, Adithya Renduchintala, Rebecca Knowles

import gzip, sys, itertools, time
import pdb
from pyspark import SparkContext, SparkConf
import nltk
from nltk.tokenize import word_tokenize
nltk.download('all')

try:
    import simplejson
except ImportError:
    import json as simplejson

import NpLayers as L
from TrainSerial import make_data, make_vocab
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

# Train the parameters of the autoencoder in bae for a maximum of max_itr
# function evaluations of l-BFGS on some partition of data in itr. Returns
# partially optimized weights.
def train(itr, bae, max_itr=1):
    ae = bae.value
    xs = []
    for x in itr:
      xs.append(x)
    (xopt, fopt, return_status) = fmin_l_bfgs_b(ae.get_cost, 
                                                ae.get_network_weights(), 
                                                ae.get_gradient, args=(xs, ),
                                                pgtol=0.1, maxfun=max_itr)
    return 1, xopt


if __name__ == '__main__':
    training_data = sys.argv[1]
    num_chunks = float(sys.argv[2])
    ae_output_fi = sys.argv[3]

    # Size of the embeddings.
    inside_width = 50

    print 'making vocab...'
    vocab_id = make_vocab(training_data, 'functionwords.txt', max_vocab=2000)
    print 'reading documents...'
    data = make_data(training_data,  vocab_id)
    print 'done reading documents', len(data), 'documents...'

    input_width = len(vocab_id)
    
    # Shrink the data size to make it tractable
    data = data[:5000]

    # This is the model we care about. The weights of this model will be updated to reflect the
    # average of multiple autoencoders (with identical topology) which are trained on different
    # subsets of the the training data.
    ae = L.Network(0.1, [input_width, inside_width, input_width])

    # Initial cost.
    prev_cost = ae.get_cost(ae.get_network_weights(), data)

    conf = (SparkConf()
         .setAppName("My app")
         .set("spark.executor.memory", "10g")
         .set("spark.storage.memoryFraction", 0.05))
    sc = SparkContext(conf=conf)

    rdd = sc.parallelize(data, numSlices=int(num_chunks))
    start_time = time.time()
    weights = None
    threshold = 3
    converged = False
    itr_converged = -1
    
    # Convergence usually occurs before 10 iterations. This should be increased
    # if input_width and inside_width are increased substantially. 
    itr = 0
    while itr < 10:
        cost = 0.0
        bae = sc.broadcast(ae)
        
        # Apply the map to the partitions of the training data
        weights = rdd.mapPartitions(lambda x: train(x, bae)).collect()

        # Sum up the weights that result from collect(). We sum up the 
        # the weights using the collect() from above and then stepping
        # through the results and adding up the weights. This appears to
        # be more stable than using a reduce which does the same thing,
        # which is why we use it here.
        summed_weights = weights[1]
        for i in range(2, len(weights)):
          # The even elements are the '1' labels, and the odd elements are the
          # weights that we want.
          if i % 2 == 0:
            continue
          summed_weights = np.add(summed_weights, weights[i])

        # Get the averaged weights and update the autoencoder.
        new_weights = summed_weights / num_chunks
        ae.set_network_weights(new_weights)
        cost = ae.get_cost(new_weights, data)
        elapsed_time = time.time() - start_time
        print "Iteration", itr, "Cost:", cost, "Prev Cost:", prev_cost, "Elapsed time (s):", elapsed_time

        if (abs(cost - prev_cost) < threshold) and not converged:
            itr_converged = itr
            converged = True
        if converged:
            print "Training has converged at iteration", itr_converged, ". Elapsed time (s):",  elapsed_time
        prev_cost = cost
        itr += 1
    L.dump(ae, ae_output_fi)
