__author__ = 'arenduchintala'

import numpy as np
import theano
import theano.tensor as T
import theano.gradient as tg


def set_value_at_position(a_location, a_value, output_model):
    print 'a_location', a_location
    print 'a_value', a_value
    print 'output_model', output_model
    zeros = T.zeros_like(output_model)
    zeros_subtensor = zeros[a_location[0], a_location[1]]
    return T.set_subtensor(zeros_subtensor, a_value)


if __name__ == '__main__':
    ""
    w1 = T.vector('w1')
    w2 = T.vector('w2')
    x = T.vector('x')
    o1_dot = T.dot(x, w1)
    o2_dot = T.dot(x, w2)
    o1 = 1.0 / (1.0 + T.exp(-o1_dot))
    o2 = 1.0 / (1.0 + T.exp(-o2_dot))
    func_o1 = theano.function([x, w1], o1)
    func_o2 = theano.function([x, w2], o2)
    g1 = T.grad(o1, w1)
    g2 = T.grad(o2, w2)
    func_g1 = theano.function([x, w1], g1)
    func_g2 = theano.function([x, w2], g2)
    print 'output layer'
    print func_o1([1, 1, 1], [1, 2, 0])
    print func_o2([1, 1, 1], [1, 1, 0])
    print 'output grad'
    print func_g1([1, 1, 1], [1, 2, 0])
    print func_g2([1, 1, 1], [1, 1, 0])
    """
    """
    # O_sum = T.sum(O)
    # func_o = theano.function([W, x], out)
    # print 'output layer mat'
    # mat_g = T.grad(O_sum, W)
    # func_matg = theano.function([W, x], mat_g)
    # print func_o(np.asarray([[1.0, 2.0, 0.0], [1.0, 1.0, 0.0]]), [1.0, 1.0, 1.0])

    print '\ntrying to get matrix vector derivatives'
    x = T.vector('x')
    W = T.matrix('W')
    out = 1.0 / (1.0 + T.exp(-T.tensordot(W, x, axes=1)))
    grad_cost, grad_updates = theano.scan(lambda i, O, W: T.grad(O[i], W[i, :]),
                                          sequences=np.asarray([0, 1]),
                                          non_sequences=[out, W])
    # func_grad = theano.function([x, W], grad_cost, updates=grad_updates)
    # print 'ouput layer grad'
    # print func_grad([1.0, 1.0, 1.0], [[1.0, 2.0, 0.0]])

    # print func_odot([[1, 2, 1], [1, 1, 0]], [1, 1, 1])
    """
    H, updates = theano.scan(lambda i, gy, x: T.grad(gy[i], x),
                             sequences=T.arange(O_dot.shape[0]),
                             non_sequences=[O_dot, x])
    f = theano.function([x], H, updates=updates)

    """
    """
    location = T.imatrix("location")
    values = T.vector("values")
    output_model = T.matrix("output_model")

    result, updates = theano.scan(fn=set_value_at_position,
                                  outputs_info=None,
                                  sequences=[location, values],
                                  non_sequences=output_model)

    assign_values_at_positions = theano.function(inputs=[location, values, output_model], outputs=result)

    # test
    test_locations = np.asarray([[1, 1], [2, 3]], dtype=np.int32)
    test_values = np.asarray([42, 50], dtype=np.float32)
    test_output_model = np.zeros((5, 5), dtype=np.float32)
    print assign_values_at_positions(test_locations, test_values, test_output_model)
    """