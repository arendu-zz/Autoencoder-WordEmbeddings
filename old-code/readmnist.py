__author__ = 'arenduchintala'
import theano
import theano.tensor as T
import numpy
import cPickle,gzip,numpy
import pdb

def convert_to_shared_format(data_x, data_y):
    #data_x is the data point and data_y is the class label
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config._floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config._floatX))
    return shared_x, T.cast(shared_y, 'int32')


if __name__ == '__main__':
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    shared_train_x, shared_train_y = covert_to_shared_format(train_set[0], train_set[1])

