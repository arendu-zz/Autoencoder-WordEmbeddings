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
    batch_weights, batch_cost = autoencoders[idx].train_earlystop(data_chunks[idx], init_weights=nn_weights, maxfun=2)
    return idx, batch_weights, batch_cost


def parallel_train_accumilate(results):
    global itr_weights, itr_cost
    print results[0], 'is accumilated'
    itr_weights += results[1]
    itr_cost += results[2]


if __name__ == '__main__':

    # script here
    corpus = "Arts.demo2.txt.gz"
    max_vocab = 5000
    num_chunks = 10
    print 'making vocab...'
    vocab_map_name = '.'.join([corpus, str(max_vocab), 'map'])
    vocab_id = make_vocab(corpus, 'functionwords.txt', max_vocab=max_vocab, save_vocab_map=vocab_map_name)
    print 'reading documents...'
    full_data = make_data(corpus, vocab_id)

    print len(vocab_id), len(full_data)

    threshold = 0.01

    # This is the model we care about. The weights of this model will be updated to reflect the
    # average of multiple autoencoders (with identical topology) which are trained on different
    # subsets of the the training data.
    avg_ae = L.Network(0.1, [len(vocab_id), 50, len(vocab_id)], full_data)
    nn_weights = avg_ae.get_network_weights()
    cpu_count = num_chunks
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
        # cost = avg_ae.get_cost(nn_weights, full_data)
        # print 'ave_ae cost:', cost
        print 'nn cost    :', nn_cost

        # if abs(cost - prev_cost) < threshold:
        # itr = 100
        # prev_cost = cost
        itr += 1

    avg_ae.set_network_weights(nn_weights)
    nn_name = '.'.join([corpus, str(max_vocab), str(num_chunks), 'nn'])
    L.dump(avg_ae, nn_name)

