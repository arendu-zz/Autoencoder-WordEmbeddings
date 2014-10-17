__author__ = 'arenduchintala'
import pdb, math
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import approx_fprime
from scipy.optimize import check_grad
import Layers

if __name__ == '__main__':
    # script here
    print '\n*********** LOGISTIC REGRESSION ************'
    x = T.vector('x')
    w = T.vector('w')
    wx = T.dot(w, x)
    y = 1.0 / (1.0 + T.exp(-wx))
    l = T.log(y) - T.dot(w, w)  # l2 regularization
    grad_l = T.grad(l, w)
    grad_y = T.grad(y, w)
    func_y = theano.function([x, w], grad_y)
    func2 = theano.function([x, w], l)
    func_g = theano.function([x, w], grad_l)
    init_w = np.array([0.0] * 5)
    init_x = np.array([0.1, 0.2, 0.3, -1.0, 3.0])
    old_cost = 0.0
    converge = False
    while not converge:
        cost = func2(init_x, init_w)
        print cost
        update = func_g(init_x, init_w)
        init_w += 0.1 * update
        if abs(cost - old_cost) < 0.0001:
            converge = True
        old_cost = cost
    print init_w

    print '\n*********** XOR GATE ************'
    data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]

    nn = Layers.Network([2,  2, 1], data)
    init_weights = nn.get_layer_weights()
    print '\nbefore training:'
    cost_nn = nn.get_cost(init_weights)
    print cost_nn
    nn.predict()

    # grad = nn.get_gradient(init_weights)

    (xopt, fopt, return_status) = fmin_l_bfgs_b(nn.get_cost, init_weights, nn.get_gradient, pgtol=0.001)
    # print xopt
    print '\nafter training:'
    print nn.get_cost(np.asarray(xopt))
    nn.predict()


