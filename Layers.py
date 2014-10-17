__author__ = 'arenduchintala'
import numpy as np
import theano
import theano.tensor as T
import math


def add_bias_input(x_input, bias=1.0):
    x_input = np.append(x_input, [bias])
    return x_input


def remove_bias(delta):
    return delta[:-1]


class HiddenLayer():
    def __init__(self, n_inputs, n_outputs, activation="softmax"):

        self.n_inputs = n_inputs + 1  # +1 for bias
        self.n_outputs = n_outputs
        ep = 0.0  # math.sqrt(6.0) / math.sqrt(self.n_inputs + self.n_outputs)
        self.W = np.random.uniform(-ep, ep, (self.n_outputs, self.n_inputs))
        # self.W = np.zeros((self.n_outputs, self.n_inputs))
        m1 = T.vector('m1')
        m2 = T.vector('m2')
        m = m1 * m2
        x = T.vector('x')
        w = T.vector('w')
        a = T.dot(w, x)
        if activation == "softmax":
            z = 1.0 / (1.0 + T.exp(-a))
            gprime = z * (1 - z)

        self.func_mult = theano.function([m1, m2], m)
        self.func_dot = theano.function([x, w], a)
        self.func_z = theano.function([x, w], z)
        self.func_gprime = theano.function([x, w], gprime)

    def weight_update(self, z_current, delta_from_next):
        delta_from_next = remove_bias(delta_from_next)
        z_current = add_bias_input(z_current)
        x1 = np.reshape(delta_from_next, (self.n_outputs, 1))
        x2 = np.reshape(z_current, (1, self.n_inputs))
        return np.multiply(x1, x2)

    def get_a(self, x_input):
        x_input = add_bias_input(x_input)
        assert np.shape(x_input) == (self.n_inputs,)
        a = np.asarray([])
        for r in xrange(np.shape(self.W)[0]):
            w_row = self.W[r]
            a = np.append(a, self.func_dot(x_input, w_row))
        assert np.shape(a) == (self.n_outputs, )
        return a

    def get_z(self, x_input):
        x_input = add_bias_input(x_input)
        assert np.shape(x_input) == (self.n_inputs, )
        z = np.asarray([])
        for r in xrange(np.shape(self.W)[0]):
            w_row = self.W[r]
            z = np.append(z, self.func_z(x_input, w_row))
        assert np.shape(z) == (self.n_outputs, )
        return z

    def get_zprime(self, x_input):
        # z = self.get_z(x_input)
        # zprime = z * (1 - z)

        x_input = add_bias_input(x_input)
        assert np.shape(x_input) == (self.n_inputs, )
        gz = np.asarray([])
        for r in xrange(np.shape(self.W)[0]):
            w_row = self.W[r]
            gz = np.append(gz, self.func_gprime(x_input, w_row))
        assert np.shape(gz) == (self.n_outputs, )
        return gz

    def get_delta(self, zprime_current, delta_from_next):
        delta_from_next = remove_bias(delta_from_next)
        zprime_current = add_bias_input(zprime_current, bias=1.0)
        assert np.shape(delta_from_next) == (self.n_outputs,)
        assert np.shape(zprime_current) == (self.n_inputs,)
        p = np.asarray([])
        for c in xrange(np.shape(self.W)[1]):
            w_col = self.W[:, c]
            p = np.append(p, self.func_dot(w_col, delta_from_next))
        delta_current = self.func_mult(p, zprime_current)
        assert np.shape(delta_current) == (self.n_inputs,)
        return delta_current

    def update(self, learning_rate, w_update):
        self.W += learning_rate * w_update


class OutputLayer(HiddenLayer):
    def __init__(self, n_inputs, n_outputs):
        HiddenLayer.__init__(self, n_inputs, n_outputs)
        z = T.vector('z')
        t = T.vector('t')
        diff = z - t
        self.func_diff = theano.function([z, t], diff)

    def get_delta_at_final(self, prediction, target_at_output):
        return self.func_diff(prediction, target_at_output)

    def get_delta(self, zprime_current, delta_from_next):
        zprime_current = add_bias_input(zprime_current, bias=1.0)
        assert np.shape(delta_from_next) == (self.n_outputs,)
        assert np.shape(zprime_current) == (self.n_inputs,)
        p = np.asarray([])
        for c in xrange(np.shape(self.W)[1]):
            w_col = self.W[:, c]
            p = np.append(p, self.func_dot(w_col, delta_from_next))
        delta_current = self.func_mult(p, zprime_current)
        assert np.shape(delta_current) == (self.n_inputs,)
        return delta_current

    def weight_update(self, z_current, delta_from_next):
        z_current = add_bias_input(z_current)
        x1 = np.reshape(delta_from_next, (self.n_outputs, 1))
        x2 = np.reshape(z_current, (1, self.n_inputs))
        return np.multiply(x1, x2)


class Network():
    def __init__(self, topology):
        self.layers = []
        for idx, (t_inp, t_out) in enumerate(zip(topology, topology[1:])):
            if idx == len(topology[1:]) - 1:
                self.layers.append(OutputLayer(t_inp, t_out))
            else:
                self.layers.append(HiddenLayer(t_inp, t_out))


    def get_cost(self, data):
        cost = 0.0
        for d, l in data[:]:
            z = d
            for idx, layer in enumerate(self.layers):
                if idx == len(self.layers) - 1:
                    # this is a output layer
                    prediction = layer.get_z(z)
                    logl = np.sum(
                        [-li * math.log(pi) - (1 - li) * math.log(1 - pi) for li, pi in zip(l, prediction)])
                    cost += logl * (1.0 / float(len(data)))
                else:
                    z = layer.get_z(z)
        return cost

    def get_gradient(self, data):
        accumulate_deltas = [np.zeros(np.shape(layer.W)) for layer in self.layers]
        for d, l in data[:]:
            z_list = [None] * (len(self.layers) + 1)
            zp_list = [None] * (len(self.layers) + 1)
            delta_list = [None] * (len(self.layers) + 1)
            z_list[0] = d
            zp_list[0] = d

            for idx, layer in enumerate(self.layers):
                z_next = layer.get_z(z_list[idx])
                z_next_prime = layer.get_zprime(zp_list[idx])
                z_list[idx + 1] = z_next
                zp_list[idx + 1] = z_next_prime

            for idx in reversed(range(len(self.layers))):
                layer = self.layers[idx]
                if isinstance(layer, OutputLayer):
                    delta = layer.get_delta_at_final(z_list[idx + 1], np.asarray(l))
                    delta_list[idx + 1] = delta
                    delta = layer.get_delta(zp_list[idx], delta_list[idx + 1])
                    delta_list[idx] = delta
                else:
                    delta = layer.get_delta(zp_list[idx], delta_list[idx + 1])
                    delta_list[idx] = delta

            for idx, layer in enumerate(self.layers):
                theta = accumulate_deltas[idx]
                theta += layer.weight_update(z_list[idx], delta_list[idx + 1]) * (1.0 / float(len(data)))
                accumulate_deltas[idx] = theta

        linear_weights = np.asarray([])
        for a in accumulate_deltas:
            length = np.shape(a)[0] * np.shape(a)[1]
            linear_weights = np.append(linear_weights, a.reshape(length, 1))

        return linear_weights


if __name__ == '__main__':
    # script here
    pass

