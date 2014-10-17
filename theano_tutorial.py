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

    theta0 = Layers.HiddenLayer(2, 2)
    theta1 = Layers.OutputLayer(2, 1)
    # theta2 = Layers.OutputLayer(3, 1)

    theta0_update = np.zeros(np.shape(theta0.W))
    theta1_update = np.zeros(np.shape(theta1.W))
    # theta2_update = np.zeros(np.shape(theta2.W))
    cost = 0.0
    learning_rate = 1.0

    for d, l in data[:]:
        print '\tinput', d, 'label', l
        z0 = np.asarray(d)
        z0_prime = np.asarray(d)

        z1 = theta0.get_z(z0)
        z1_prime = theta0.get_zprime(z0)

        z2 = theta1.get_z(z1)
        z2_prime = theta1.get_zprime(z1)

        # z3 = theta2.get_z(z2)
        # z3_prime = theta2.get_zprime(z2)

        prediction = z2

        # d3 = theta2.get_delta_at_final(z3, np.asarray(l))
        d2 = theta1.get_delta_at_final(z2, np.asarray(l))
        # d2 = theta2.get_delta(z2_prime, d3)
        d1 = theta1.get_delta(z1_prime, d2)

        logl = np.sum([-li * math.log(pi) - (1 - li) * math.log(1 - pi) for li, pi in zip(l, prediction)])
        cost += logl * (1.0 / float(len(data)))

        # theta2_update += theta2.weight_update(z2, d3) * (1.0 / float(len(data)))
        theta1_update += theta1.weight_update(z1, d2) * (1.0 / float(len(data)))
        theta0_update += theta0.weight_update(z0, d1) * (1.0 / float(len(data)))
        # print 't1 update', theta1_update
        # print 't2 update', theta2_update
    print cost

    layers = [theta0, theta1]
    accumulate_deltas = [np.zeros(np.shape(layer.W)) for layer in layers]
    for d, l in data[:]:
        z_list = [None] * (len(layers) + 1)
        zp_list = [None] * (len(layers) + 1)
        delta_list = [None] * (len(layers) + 1)
        z_list[0] = d
        zp_list[0] = d

        for idx, layer in enumerate(layers):
            z_next = layer.get_z(z_list[idx])
            z_next_prime = layer.get_zprime(zp_list[idx])
            z_list[idx + 1] = z_next
            zp_list[idx + 1] = z_next_prime

        for idx in reversed(range(len(layers))):
            layer = layers[idx]
            if isinstance(layer, Layers.OutputLayer):
                delta = layer.get_delta_at_final(z_list[idx + 1], np.asarray(l))
                delta_list[idx + 1] = delta
                delta = layer.get_delta(zp_list[idx], delta_list[idx + 1])
                delta_list[idx] = delta
            else:
                delta = layer.get_delta(zp_list[idx], delta_list[idx + 1])
                delta_list[idx] = delta

        for idx, layer in enumerate(layers):
            theta = accumulate_deltas[idx]
            theta += layer.weight_update(z_list[idx], delta_list[idx + 1]) * (1.0 / float(len(data)))
            accumulate_deltas[idx] = theta

    linear_weights = np.asarray([])
    for a in accumulate_deltas:
        length = np.shape(a)[0] * np.shape(a)[1]
        linear_weights = np.append(linear_weights, a.reshape(length, 1))

    print 'ok'

    nn = Layers.Network([2, 2, 1])
    cost_nn = nn.get_cost(data)
    grad = nn.get_gradient(data)
    print np.array_str(grad)
    print np.array_str(linear_weights)
    print 'ok'





