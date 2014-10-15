__author__ = 'arenduchintala'
import numpy as np
import theano
import theano.tensor as T


class HiddenUnit():
    def __init__(self, id, n_outputs):
        pass


class HiddenLayer():
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = np.asarray(np.zeros((n_outputs, n_inputs)))
        x = T.vector('x')
        w = T.vector('w')
        m = x * w
        self.func_mult = theano.function([x, w], m)
        dot = T.dot(x, w)
        self.func_dot = theano.function([x, w], dot)
        z = 1.0 / (1.0 + T.exp(-dot))
        self.func_z = theano.function([x, w], z)
        gz = T.grad(z, w)
        self.func_gz = theano.function([x, w], gz)

    def get_a(self, x_input):
        a = [None] * self.n_outputs
        for r in xrange(np.shape(self.W)[0]):
            w_row = self.W[r]
            a[r] = self.func_dot(x_input, w_row)
        return np.asarray(a)

    def get_z(self, x_input):
        z = [None] * self.n_outputs
        for r in xrange(np.shape(self.W)[0]):
            w_row = self.W[r]
            z[r] = self.func_z(x_input, w_row)
        return np.asarray(z)

    def get_gradz(self, x_input):
        grad_z = [None] * self.n_outputs
        for r in xrange(np.shape(self.W)[0]):
            w_row = self.W[r]
            grad_z[r] = self.func_gz(x_input, w_row)
        return np.asarray(grad_z)

    def get_delta(self, grad_at_input, delta_at_output):
        delta = [None] * self.n_inputs
        for c in xrange(np.shape(self.W)[1]):
            w_col = self.W[:, c]
            delta[c] = self.func_dot(delta_at_output, w_col)
        return self.func_mult(delta_at_output, grad_at_input)


class InputLayer(HiddenLayer):
    def __init__(self, n_inputs):
        pass


class OutputLayer(HiddenLayer):
    def __init__(self, n_inputs, n_outputs):
        pass


if __name__ == '__main__':
    # script here
    pass

