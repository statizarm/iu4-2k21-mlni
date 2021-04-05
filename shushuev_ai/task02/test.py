import unittest
import numpy as np
import neuralnet
from neuralnet import Layer


class Test(unittest.TestCase):
    def test_create_layer(self):
        expected = np.array([[0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0]])

        received = Layer(expected)

        self.assert_((expected == received.w).all(), f'expected={expected}\nreceived={received.w}\n')

    def test_layer_with_zero_weights_forward(self):
        w = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]])

        expected = np.array([1/3, 1/3, 1/3])

        l = Layer(w)

        received = l.forward(np.array([1, 2, 3]))

        self.assert_((expected == received).all(), f'expected={expected}\nreceived={received}')

