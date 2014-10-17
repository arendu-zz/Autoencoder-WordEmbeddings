__author__ = 'arenduchintala'
import pdb, math
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import Layers

if __name__ == '__main__':
    # script here
    """
    x = T.scalar('x')
    y = x ** 2
    print x
    print y
    print type(y)
    print y.type
    print theano.pp(y)

    func = theano.function([x], y)
    print func(2.0)

    # maximize a function y = -x^2 _ 2x

    x = T.scalar('x')
    y = -(x ** 2) + (2 * x)
    func1 = theano.function([x], y)
    grad1 = T.grad(y, x)
    gy = theano.function([x], grad1)
    init_x = 0.0
    delta_x = 10
    while delta_x > 0.001:
        print 'x:', init_x
        print 'y:', func1(init_x)
        delta_x = gy(init_x)
        print 'gy:', delta_x
        init_x = init_x + (0.01) * delta_x
    print 'final x:', init_x
    print 'final y:', func1(init_x)

    """
    # logistic regression

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




    # xor gate
    data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]

    theta1 = Layers.HiddenLayer(2, 2)
    theta2 = Layers.HiddenLayer(2, 3)
    theta3 = Layers.OutputLayer(3, 1)

    theta1_update = np.zeros(np.shape(theta1.W))
    theta2_update = np.zeros(np.shape(theta2.W))
    theta3_update = np.zeros(np.shape(theta3.W))
    cost = 0.0
    learning_rate = 1.0
    for iter in xrange(1):
        print '\niteration', iter
        for d, l in data[:]:
            print '\tinput', d, 'label', l
            z1 = np.asarray(d)
            z2 = theta1.get_z(z1)
            z2_prime = theta1.get_zprime(z1)

            z3 = theta2.get_z(z2)
            z3_prime = theta2.get_zprime(z2)

            z4 = theta3.get_z(z3)
            z4_prime = theta3.get_zprime(z3)

            d4 = theta3.get_delta_at_final(z4, np.asarray(l))
            d3 = theta3.get_delta(z3_prime, d4)
            d2 = theta2.get_delta(z2_prime, d3)

            logl = np.sum([-li * math.log(z3i) - (1 - li) * math.log(1 - z3i) for li, z3i in zip(l, z3)])
            cost += logl * (1.0 / float(len(data)))

            theta3_update += theta3.weight_update(z3, d4) * (1.0 / float(len(data)))
            theta2_update += theta2.weight_update(z2, d3) * (1.0 / float(len(data)))
            theta1_update += theta1.weight_update(z1, d2) * (1.0 / float(len(data)))
            # print 't1 update', theta1_update
            # print 't2 update', theta2_update
        print cost




