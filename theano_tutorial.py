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
        init_w = init_w + (0.1) * update
        if abs(cost - old_cost) < 0.0001:
            converge = True
        old_cost = cost
    print init_w

    x = T.vector('x')
    w = T.vector('w')
    y = T.dot(x, w)  # T.tensordot(x, w, axes=1)
    funcy = theano.function([w, x], y)
    # J, updates = theano.scan(lambda i, y, w: T.grad(y[i], w), sequences=T.arange(y.shape[0]), non_sequences=[y, w])
    # f = theano.function([w, x], J, updates=updates)
    grady = T.grad(y, w)
    f = theano.function([w, x], grady)
    mat = np.asarray([[1, 2, 3], [2, 2, 2], [2, 1, 4]])
    vec = np.asarray([1, 2, 1])
    print 'cost', funcy(vec, vec)
    print 'grad', f(vec, [1, 1, 1])

    data = [([0, 0], [0, 0]), ([0, 1], [1, 1]), ([1, 0], [1, 1]), ([1, 1], [0, 0])]
    x = T.vector('x')
    w = T.matrix('w')
    h = T.tensordot(x, w, axes=1)
    sm = 1.0 / (1.0 + T.exp(-h))
    func_prod = theano.function([x, w], sm)
    W_hidden = np.asarray(np.zeros((2, 2)), dtype=theano.config.floatX)
    X_values = np.asarray([1, 1], dtype=theano.config.floatX)
    # print np.shape(W_hidden), W_hidden

    # print np.shape(X_values), X_values
    # print func_prod(X_values, W_hidden)

    il = Layers.HiddenLayer(2, 3)
    # hl = Layers.HiddenLayer(3, 2)
    ol = Layers.OutputLayer(3, 2)

    for d, l in data[:]:
        print 'input', d, 'label', l
        il_output = il.get_z(np.asarray(d))
        # hl_output = hl.get_z(il_output)
        ol_output = ol.get_z(il_output)
        ol_delta_final = ol.get_delta_at_final(il_output, np.asarray(l))
        print 'prediction_ol:', ol_output, 'ol_delta_final:', ol_delta_final
        ol_delta = ol.get_delta(il_output, ol_delta_final)
        print 'ol_delta:', ol_delta
        # ol_delta = ol.get_delta(il_output, ol_delta)
        # print 'hl_delta', hl_delta
        # il_delta = il.get_delta(np.asarray(d), hl_delta)
        # print 'il_delta', il_delta

    # hl_output = hl.get_output(il_output)
    # prediction = ol.get_output(hl_output)
    # print 'prediction', prediction
    # print 'grad at output', ol.get_grad(hl_output)
    # o_delta = ol.get_delta(hl_output, np.asarray(l))
    # print 'o delta', o_delta
    # h_delta = hl.get_delta(il_output, o_delta)
    # print 'h delta', h_delta

    # inp_val = il.get_inp(d)
    # print 'inp val', inp_val

    # hl = Layers.HiddenLayer(2, 2)
    # out_hidden = hl.get_output(inp_val)
    # print 'hidden', out_hidden
    """
    ol = Layers.OutputLayer(2, 1)
    pred = ol.get_prediction(out_hidden)
    print 'pred', pred, 'true', l
    print 'error', ol.get_delta(out_hidden, np.asarray(l))
    print 'grad', ol.get_grad(out_hidden)
    """
