__author__ = 'arenduchintala'
import gzip
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process
import numpy as np

try:
    import simplejson
except ImportError:
    import json as simplejson

import NpLayers as L
from ReadAmazonReview import make_data, make_vocab

global itr_weights, itr_cost, num_chunks, autoencoders, data_chunks
itr_cost = 0.0
itr_weights = np.zeros(9)
num_chunks = 2
autoencoders = []
data_chunks = []


def parallel_train(idx, nn_weights, data_chunk):
    global autoencoders, data_chunks
    print 'training batch', idx
    print autoencoders[idx].get_cost(nn_weights)
    # batch_weights, batch_cost = autoencoders[idx].train_earlystop(data_chunks[idx], init_weights=nn_weights, maxfun=2)
    # return idx, batch_weights, batch_cost
    return idx  # , np.zeros(10), 1.0


def parallel_train_accumilate(results):
    global itr_weights, itr_cost
    print('in acc')
    # itr_weights += results[1]
    # itr_cost += results[2]
    print 'accumilated batch', results


if __name__ == '__main__':

    # script here
    print 'making vocab...'
    vocab_id = make_vocab('Arts.demo2.txt.gz', 'functionwords.txt', max_vocab=2000)
    print 'reading documents...'
    full_data = make_data('Arts.demo2.txt.gz', vocab_id)

    # full_data = full_data[:500]
    print len(vocab_id), len(full_data)
    # simulated parallel

    threshold = 0.01
    converged = False


    # This is the model we care about. The weights of this model will be updated to reflect the
    # average of multiple autoencoders (with identical topology) which are trained on different
    # subsets of the the training data.
    avg_ae = L.Network(0.1, [len(vocab_id), 50, len(vocab_id)], full_data)
    nn_weights = avg_ae.get_network_weights()

    num_chunks = 4.0

    for c in xrange(int(num_chunks)):
        data_chunk = full_data[c * int(len(full_data) / num_chunks): (c + 1) * int(len(full_data) / num_chunks)]
        ae = L.Network(0.1, [len(vocab_id), 50, len(vocab_id)], data_chunk)
        ae.set_network_weights(nn_weights)
        data_chunks.append(data_chunk)
        autoencoders.append(ae)
    print 'initialized parallel encoders...'

    prev_cost = avg_ae.get_cost(nn_weights, full_data)
    print 'initial ave_ae cost', prev_cost
    itr = 0
    while itr < 10:
        itr_cost = 0.0
        itr_weights = np.zeros(np.shape(nn_weights))
        """
        for idx, ae in enumerate(autoencoders):
            print 'chunk', idx
            dc = data_chunks[idx]
            batch_weights, batch_cost = ae.train_earlystop(dc, init_weights=nn_weights, maxfun=2)
            itr_weights += batch_weights
            itr_cost += batch_cost
        """
        cpu_count = multiprocessing.cpu_count()
        pool = Pool(processes=cpu_count)
        for idx, ae in enumerate(autoencoders):
            dc = data_chunks[idx]
            # p = Process(target=parallel_train, args=(idx, ae, nn_weights, dc), callback=parallel_train_accumilate)
            pool.apply_async(parallel_train, args=(idx, nn_weights, dc),
                             callback=parallel_train_accumilate)
        pool.close()
        pool.join()

        nn_weights = (1.0 / num_chunks) * itr_weights  # average_weights(autoencoders)
        nn_cost = (1.0 / num_chunks) * itr_cost
        cost = avg_ae.get_cost(nn_weights, full_data)
        print 'ave_ae cost:', cost
        print 'nn cost    :', nn_cost

        if abs(cost - prev_cost) < threshold:
            itr = 100
        prev_cost = cost
        itr += 1

    avg_ae.set_network_weights(nn_weights)
    L.dump(avg_ae, 'average-ae-Arts-2000.nn')

