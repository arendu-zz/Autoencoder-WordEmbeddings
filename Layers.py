__author__ = 'arenduchintala'
import numpy as np
import theano
import theano.tensor as T


class HiddenUnit():
    def __init__(self, id, n_outputs, activation="Softmax"):
        self.id = id
        self.n_outputs = n_outputs
        self.W_instance = np.asarray([0] * self.n_outputs)

        gx = T.vector('gx')
        gy = T.vector('gy')
        dgxgy = T.dot(gx, gy)
        self.dot_func = theano.function([gx, gy], dgxgy)
        x = T.vector('x')
        weights = T.vector('weights')
        wx = T.vector('wx')
        if activation == "Softmax":
            output = 1.0 / (1.0 + T.exp(-wx))
        elif activation is None:
            output = wx

        grad_output = T.grad(output, weights)
        self.output_func = theano.function([x, weights], output)
        self.grad_func = theano.function([x, weights], grad_output)


    def init_weight(self, w):
        self.W_instance = w

    def get_output(self, x_instance):
        return self.output_func(x_instance, self.W_instance)

    def get_grad(self, x_instance):
        return self.grad_func(x_instance, self.W_instance)

    def outgoing_weights(self, nextLayer):
        self.W_out = [None] * len(nextLayer.hidden_units)
        for i_nhu, nhu in enumerate(nextLayer.hidden_units):
            for i_w_out, w_out in enumerate(nhu.W_instance):
                if i_w_out == self.id:
                    self.W_out[i_nhu] = w_out
        self.W_out = np.asarray(self.W_out)

    def get_delta(self, x_instance, delta_from_nextlayer):
        hu_delta = self.dot_func(self.W_out, delta_from_nextlayer) * self.get_grad(x_instance)
        return hu_delta

    def update_weights(self, delta_weights):
        pass


class HiddenLayer():
    def __init__(self, n_inputs, n_outputs, activation="Softmax"):
        self.n_inputs = n_inputs
        self.n_ouputs = n_outputs
        self.activation = activation
        gx = T.vector('gx')
        gy = T.vector('gy')
        dgxgy = T.dot(gx, gy)
        self.dot_func = theano.function([gx, gy], dgxgy)
        print 'dot func', self.dot_func([1, 2, 4], [3, 1, 1])
        self.hidden_units = []
        for i in range(n_outputs):
            hu = HiddenUnit(id=i, n_outputs=n_inputs, activation=activation)
            self.hidden_units.append(hu)

    def get_output(self, x_instance):
        o = [h.get_output(x_instance) for h in self.hidden_units]
        return np.asarray(o)

    def merge_layer(self, nextLayer):
        for hu in self.hidden_units:
            hu.outgoing_weights(nextLayer)

    def get_grad(self, x_instance):
        print self.activation
        for h in self.hidden_units:
            print h.get_grad(x_instance)
        g = [h.get_grad(x_instance) for h in self.hidden_units]
        return np.asarray(g)

    def get_delta(self, x_instance, delta_from_nextlayer):
        delta = [None] * len(self.hidden_units)
        for ihu, hu in enumerate(self.hidden_units):
            delta[ihu] = hu.get_delta(x_instance, delta_from_nextlayer)
        return np.asarray(delta)


class InputLayer(HiddenLayer):
    def __init__(self, n_inputs):
        HiddenLayer.__init__(self, n_inputs, n_inputs, None)
        self.init_input_weights()

    def init_input_weights(self):
        # for an input layer the input matrix W is Identity
        eye = np.eye(self.n_inputs)
        for idx, r in enumerate(eye):
            # print idx, r
            self.hidden_units[idx].init_weight(r)


class OutputLayer(HiddenLayer):
    def __init__(self, n_inputs, n_outputs):
        HiddenLayer.__init__(self, n_inputs, n_outputs)

    def get_grad(self, x_instance):
        return [0.1]

    def get_delta(self, x_instance, target):
        pred = self.get_output(x_instance)
        diff = target - pred
        return diff


if __name__ == '__main__':
    # script here
    pass

