__author__ = 'arenduchintala'
__author__ = 'arenduchintala'
import numpy as np
import math
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import fmin_bfgs

np.set_printoptions(precision=2, suppress=True)


def sigmoid(a):
    d = - a
    d = 1.0 + np.exp(d)
    d = 1.0 / d
    return d.reshape(np.shape(a))


def sigmoid_prime(z):
    return z * (1.0 - z)


def add_bias_input(vec, bias=1.0):
    vec = np.append(vec, [bias])
    return vec.reshape(np.size(vec), 1)


def remove_bias(delta):
    return delta[:-1]


def safe_log(l, p):
    if abs(l - p) == 1.0:
        return -100.0
    elif l == 1:
        return math.log(p)
    else:
        return math.log(1.0 - p)


class HiddenLayer():
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs + 1  # +1 for bias
        self.n_outputs = n_outputs
        ep = math.sqrt(6.0) / math.sqrt(self.n_inputs + self.n_outputs)
        self.W = np.random.uniform(-ep, ep, (self.n_outputs, self.n_inputs))
        # self.W = np.zeros((self.n_outputs, self.n_inputs))


    def func_dot(self, x, w):
        assert np.shape(x) == np.shape(w)
        return np.dot(x.T, w)[0, 0]

    def func_mult(self, m1, m2):
        assert np.shape(m1) == np.shape(m2)
        return m1 * m2

    def func_z(self, x, w):
        a = self.func_dot(x, w)
        z = sigmoid(a)
        return z

    def func_gprime(self, x, w):
        z = self.func_z(x, w)
        gz = sigmoid_prime(z)
        return gz

    def weight_update(self, z_current, delta_from_next):
        delta_from_next = remove_bias(delta_from_next)
        z_current = add_bias_input(z_current)
        x1 = np.reshape(delta_from_next, (self.n_outputs, 1))
        x2 = np.reshape(z_current, (1, self.n_inputs))
        return np.multiply(x1, x2)

    def get_a(self, x_input):
        x_input = add_bias_input(x_input)
        assert np.shape(x_input) == (self.n_inputs, 1)
        a = np.dot(self.W, x_input)
        assert np.shape(a) == (self.n_outputs, 1)
        return a

    def get_z(self, x_input):
        assert np.shape(x_input) == (self.n_inputs - 1, 1)
        x_input = add_bias_input(x_input)
        assert np.shape(x_input) == (self.n_inputs, 1)
        a = np.dot(self.W, x_input)
        z = sigmoid(a)
        assert np.shape(z) == (self.n_outputs, 1)
        return z

    def get_zprime(self, x_input):
        x_input = add_bias_input(x_input)
        assert np.shape(x_input) == (self.n_inputs, 1)
        a = np.dot(self.W, x_input)
        z = sigmoid(a)
        gz = sigmoid_prime(z)
        assert np.shape(gz) == (self.n_outputs, 1)
        return gz

    def get_delta(self, zprime_current, delta_from_next):
        delta_from_next = remove_bias(delta_from_next)
        zprime_current = add_bias_input(zprime_current, bias=1.0)
        assert np.shape(delta_from_next) == (self.n_outputs, 1)
        assert np.shape(zprime_current) == (self.n_inputs, 1)
        p_test = np.dot(self.W.T, delta_from_next)
        delta_current = p_test * zprime_current
        assert np.shape(delta_current) == (self.n_inputs, 1)
        return delta_current

    def update(self, learning_rate, w_update):
        self.W += learning_rate * w_update


class OutputLayer(HiddenLayer):
    def __init__(self, n_inputs, n_outputs):
        HiddenLayer.__init__(self, n_inputs, n_outputs)

    def get_delta_at_final(self, prediction, target_at_output):
        assert np.shape(prediction) == (self.n_outputs, 1)
        assert np.shape(target_at_output) == (self.n_outputs, 1)
        return prediction - target_at_output

    def get_delta(self, zprime_current, delta_from_next):
        zprime_current = add_bias_input(zprime_current, bias=1.0)
        assert np.shape(delta_from_next) == (self.n_outputs, 1)
        assert np.shape(zprime_current) == (self.n_inputs, 1)
        p_test = np.dot(self.W.T, delta_from_next)
        delta_current = p_test * zprime_current
        assert np.shape(delta_current) == (self.n_inputs, 1)
        return delta_current

    def weight_update(self, z_current, delta_from_next):
        z_current = add_bias_input(z_current)
        x1 = np.reshape(delta_from_next, (self.n_outputs, 1))
        x2 = np.reshape(z_current, (1, self.n_inputs))
        return x1 * x2


class Network():
    def __init__(self, lmbda, topology, data):
        self.layers = []
        self.N = float(len(data))
        self.lmbda = lmbda
        self.data = data
        for idx, (t_inp, t_out) in enumerate(zip(topology, topology[1:])):
            if idx == len(topology[1:]) - 1:
                self.layers.append(OutputLayer(t_inp, t_out))
            else:
                self.layers.append(HiddenLayer(t_inp, t_out))

    def predict(self, scale=False):
        predictions = []
        for d, l in self.data[:]:
            z = d
            for idx, layer in enumerate(self.layers):
                if idx == len(self.layers) - 1:
                    # this is a output layer
                    prediction = layer.get_z(z)
                    predictions.append(prediction)
                    if scale:
                        r = np.max(prediction) - np.min(prediction)
                        p = prediction - np.min(prediction)
                        pminmax = p * (1.0 / r)
                        x = zip(l, pminmax)
                        x = np.asarray(x)
                    else:
                        x = np.hstack((l, prediction))
                    print '\ndata:', d.T
                    print 'label', l.T
                    print 'prediction:', prediction.T
                else:
                    z = layer.get_z(z)
        return predictions

    def get_layer_weights(self):
        linear_weights = np.asarray([])
        for l in self.layers:
            length = np.shape(l.W)[0] * np.shape(l.W)[1]
            linear_weights = np.append(linear_weights, l.W.reshape(length, 1))
        return linear_weights

    def set_layer_weights(self, weights):
        st = 0
        for l in self.layers:
            end = st + (np.shape(l.W)[0] * np.shape(l.W)[1])
            segment = weights[st:end]
            new_w = segment.reshape(np.shape(l.W))
            l.W = new_w
            st = end


    def get_cost(self, weights):
        # print 'getting cost...'
        # reg = (self.lmbda / 2.0 * self.N) * np.sum(weights ** 2)
        reg = (self.lmbda / self.N) * np.sum(np.abs(weights))

        self.set_layer_weights(weights)

        cost = 0.0
        for d, l in self.data[:]:
            z = d
            for idx, layer in enumerate(self.layers):
                if idx == len(self.layers) - 1:
                    # this is a output layer
                    prediction = layer.get_z(z)
                    logl = np.sum(
                        [-li * safe_log(li, pi) - (1 - li) * safe_log(li, pi) for li, pi in zip(l, prediction)])
                    cost += logl * (1.0 / float(self.N))
                else:
                    z = layer.get_z(z)
        return cost + reg

    def get_gradient(self, weights):
        # print 'getting gradient...'
        reg = (self.lmbda / self.N) * weights
        self.set_layer_weights(weights)
        accumulate_deltas = [np.zeros(np.shape(layer.W)) for layer in self.layers]
        for d, l in self.data[:]:
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
                theta += layer.weight_update(z_list[idx], delta_list[idx + 1]) * (1.0 / float(self.N))
                accumulate_deltas[idx] = theta

        linear_deltas = np.asarray([])
        for a in accumulate_deltas:
            length = np.shape(a)[0] * np.shape(a)[1]
            linear_deltas = np.append(linear_deltas, a.reshape(length, 1))
        linear_deltas += reg
        return linear_deltas


if __name__ == '__main__':
    # script here
    data = [([0, 1], [1]),
            ([0, 0], [0]),
            ([1, 0], [1]),
            ([1, 1], [0])]
    data = [(np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)))
            for x, y in data]

    nn = Network(0.0001, [2, 2, 1], data)
    init_weights = nn.get_layer_weights()
    print '\nbefore training:'
    cost_nn = nn.get_cost(init_weights)
    print cost_nn
    nn.predict()

    grad = nn.get_gradient(init_weights)
    print grad
    (xopt, fopt, return_status) = fmin_bfgs(nn.get_cost, init_weights, nn.get_gradient, pgtol=0.0001)
    # print xopt
    print '\nafter training:'
    print nn.get_cost(np.asarray(xopt))
    nn.predict()
