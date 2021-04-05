import numpy as np


class Layer(object):
    def __init__(self, w: np.array):
        self.w = w

    def forward(self, x: np.array):
        return self._soft_max(self.w @ x)

    def backward(self, x: np.array, delta: np.array):
        pass

    @staticmethod
    def _soft_max(z: np.array) -> np.array:
        a = np.exp(z)
        return a / np.sum(a, axis=0)
