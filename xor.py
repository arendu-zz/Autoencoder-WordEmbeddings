__author__ = 'arenduchintala'
import Layers
import numpy as np
import math

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
