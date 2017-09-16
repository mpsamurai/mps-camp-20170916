import numpy as np


class Neuron:
    def __init__(self):
        self.w = np.random.rand(2)  # [ , ] 2次元
        self.b = np.random.rand(1)  # [] 1次元

    def neuron(self, x):
        s0 = (x.dot(self.w)) + 1 * self.b
        return s0

    def output(self, s0):
        y = 1 / (1 + np.exp(-s0))
        return y


if __name__ == '__main__':
    n = Neuron()
    x = np.array([1., 2.])
    i = n.neuron(x)
    print(n.output(i))

