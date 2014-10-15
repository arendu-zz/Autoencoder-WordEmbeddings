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
        mx = T.vector('mx')
        mw = T.vector('mw')
        m = mx * mw
        x = T.vector('x')
        w = T.vector('w')
        dot = T.dot(w, x)
        z = 1.0 / (1.0 + T.exp(-dot))
        gz = T.grad(z, w)
        self.func_mult = theano.function([mx, mw], m)
        self.func_dot = theano.function([x, w], dot)
        self.func_gz = theano.function([x, w], gz)
        self.func_z = theano.function([x, w], z)


    def get_a(self, x_input):
        a = np.asarray([])
        for r in xrange(np.shape(self.W)[0]):
            w_row = self.W[r]
            a = np.append(a, self.func_dot(x_input, w_row))
        return a

    def get_z(self, x_input):
        z = np.asarray([])
        for r in xrange(np.shape(self.W)[0]):
            w_row = self.W[r]
            z = np.append(z, self.func_z(x_input, w_row))
        return z

    def get_gradz(self, x_input):
        grad_z = None
        for r in xrange(np.shape(self.W)[0]):
            w_row = self.W[r]
            gz = self.func_gz(x_input, w_row)
            gz = gz.reshape(1, np.shape(self.W)[1])
            if grad_z is not None:
                grad_z = np.append(grad_z, gz, axis=0)
            else:
                grad_z = gz
        return grad_z

    def get_delta(self, x_input, delta_from_next):
        grad_current = self.get_gradz(x_input)
        # TODO: figure out this multiplication properly
        delta = np.asarray([])
        for c in xrange(np.shape(self.W)[1]):
            w_col = self.W[:, c]
            temp = self.func_dot(delta_from_next, w_col)
            delta = np.append(delta, temp)
        delta_current = self.func_mult(delta, grad_current)
        return delta_current


"""
class InputLayer(HiddenLayer):
    def __init__(self, n_inputs, n_outputs):
        HiddenLayer.__init__(self, n_inputs, n_outputs)

    def get_delta(self):
        return np.asarray([0.0] * self.n_inputs)
"""


class OutputLayer(HiddenLayer):
    def __init__(self, n_inputs, n_outputs):
        HiddenLayer.__init__(self, n_inputs, n_outputs)
        a = T.vector('a')
        t = T.vector('t')
        diff = a - t
        self.func_diff = theano.function([a, t], diff)

    def get_delta_at_final(self, x_inputs, target_at_output):
        a = self.get_a(x_inputs)
        return self.func_diff(a, target_at_output)


if __name__ == '__main__':
    # script here
    pass

