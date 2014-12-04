import numpy as np
import math
from scipy.optimize import minimize
import cPickle as pickle

np.set_printoptions(precision=2, suppress=True)


def dump(nn, location):
    f = file(location, 'wb')
    pickle.dump(nn, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def load(location):
    f = file(location, 'rb')
    nn = pickle.load(f)
    f.close()
    return nn


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
    def __init__(self, lmbda=None, topology=None, data=None):
        self.layers = []
        self.lmbda = lmbda
        # self.data = data
        if topology is not None:
            self.topology = topology
            self.layers = self.make_layers(topology)
        if data is not None:
            self.N = float(len(data))

    def size_bytes(self):
      tot_bytes = 0
      for l in self.layers:
        tot_bytes += l.W.nbytes
      return tot_bytes

    def make_layers(self, topology):
        layers = []
        for idx, (t_inp, t_out) in enumerate(zip(topology, topology[1:])):
            if idx == len(topology[1:]) - 1:
                # self.layers.append(OutputLayer(t_inp, t_out))
                layers.append(OutputLayer(t_inp, t_out))
            else:
                # self.layers.append(HiddenLayer(t_inp, t_out))
                layers.append(HiddenLayer(t_inp, t_out))
        return layers


    def predict(self, data, scale=False):
        predictions = []
        for d, l in data[:]:
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
                else:
                    z = layer.get_z(z)
        return predictions

    def get_representation(self, x_input, layer=0):
        """
        :returns output of the layer (after sigmoid)
        :param x_input:
        :param layer:
        :return:
        """
        z = self.layers[layer].get_z(x_input)
        return z

    def get_network_weights(self):
        linear_weights = np.asarray([])
        for l in self.layers:
            length = np.shape(l.W)[0] * np.shape(l.W)[1]
            linear_weights = np.append(linear_weights, l.W.reshape(length, 1))
        return linear_weights

    def set_network_weights(self, weights):
        st = 0
        for l in self.layers:
            end = st + (np.shape(l.W)[0] * np.shape(l.W)[1])
            segment = weights[st:end]
            new_w = segment.reshape(np.shape(l.W))
            l.W = new_w
            st = end

    def convert_weights_to_layers(self, weights):
        layers = self.make_layers(self.topology)
        st = 0
        for l in layers:
            end = st + (np.shape(l.W)[0] * np.shape(l.W)[1])
            segment = weights[st:end]
            new_w = segment.reshape(np.shape(l.W))
            l.W = new_w
            st = end
        return layers


    def get_cost(self, weights, data, display=False):
        # print 'getting cost...'
        N = float(len(data))
        reg = (self.lmbda / (2.0 * N)) * np.sum(weights ** 2)
        # reg = (self.lmbda / self.N) * np.sum(np.abs(weights))
        # self.set_network_weights(weights)
        layers = self.convert_weights_to_layers(weights)
        cost = 0.0
        for d, l in data[:]:
            z = d
            for idx, layer in enumerate(layers):
                if idx == len(layers) - 1:
                    # this is a output layer
                    prediction = layer.get_z(z)
                    prediction[prediction >= 1.0] = 1.0 - np.finfo(float).eps  # to avoid nan showing up
                    prediction[prediction <= 0.0] = 0.0 + np.finfo(float).eps
                    l1p = -l * np.log(prediction)
                    l2p = -(1.0 - l) * np.log((1.0 - prediction))
                    lcost = np.sum(l1p + l2p)
                    cost += lcost * (1.0 / float(N))
                else:

                    z = layer.get_z(z)
        if display:
            print 'cost', cost + reg
        return cost + reg

    def get_gradient(self, weights, data, display=False):
        # print 'getting grad...'
        N = float(len(data))
        reg = (self.lmbda / N) * weights
        # self.set_network_weights(weights)
        layers = self.convert_weights_to_layers(weights)
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
                if isinstance(layer, OutputLayer):
                    delta = layer.get_delta_at_final(z_list[idx + 1], np.asarray(l))
                    delta_list[idx + 1] = delta
                    delta = layer.get_delta(zp_list[idx], delta_list[idx + 1])
                    delta_list[idx] = delta
                else:
                    delta = layer.get_delta(zp_list[idx], delta_list[idx + 1])
                    delta_list[idx] = delta

            for idx, layer in enumerate(layers):
                theta = accumulate_deltas[idx]
                N = len(data)
                theta += layer.weight_update(z_list[idx], delta_list[idx + 1]) * (1.0 / float(N))
                accumulate_deltas[idx] = theta

        linear_deltas = np.asarray([])
        for a in accumulate_deltas:
            length = np.shape(a)[0] * np.shape(a)[1]
            linear_deltas = np.append(linear_deltas, a.reshape(length, 1))
        linear_deltas += reg
        return linear_deltas

    def train(self, data):
        t1 = minimize(self.get_cost, init_weights, method='L-BFGS-B', jac=self.get_gradient,
                      args=(data, ),
                      tol=0.000001)
        return t1.x


import utils

if __name__ == '__main__':
    # script here
    data = [([0, 1], [1]),
            ([0, 0], [0]),
            ([1, 0], [1]),
            ([1, 1], [0])]
    data = [(np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)))
            for x, y in data]

    nn = Network(0.0001, [2, 2, 1], data)
    init_weights = nn.get_network_weights()
    grad = nn.get_gradient(init_weights, data)

    grad_approx = utils.gradient_checking(init_weights, 1e-4, nn.get_cost, data)
    print 'cosine similarity between grad and finite difference approx', utils.cosine_sim(grad, grad_approx)

    nn = Network(0.0001, [2, 2, 1], data)
    init_weights = nn.get_network_weights()
    print 'before training:', nn.get_cost(init_weights, data)
    final_weights = nn.train(data)
    print 'after training:', nn.get_cost(np.asarray(final_weights), data)
    nn.set_network_weights(final_weights)
    p = [round(float(l.item()), 2) for l in nn.predict(data)]
    print 'prediction:', p
    l = [float(l) for d, l in data]
    print 'label     :', l
    print 'cosine    :', utils.cosine_sim(np.array(p), np.array(l))

    nn.set_network_weights(final_weights)
    dump(nn, 'test')
    nn1 = Network()
    nn1 = load('test')
    print '\nafter pickle'
    print nn1.get_cost(final_weights, data)
