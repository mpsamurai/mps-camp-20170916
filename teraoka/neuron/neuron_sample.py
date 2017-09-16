class Neuron:
    def __init__(self, n):
        self.w = np.random.rand(n)  # setされた次元数
        self.b = np.random.rand(1)  # [] 1次元の要素をバイアスに与える

    def set_state(self, x):
        s0 = (x.dot(self.w)) + 1 * self.b
        return s0

    def activate(self, s0):
        y = 1 / (1 + np.exp(-s0))
        return y

    def output(self, x):
        return self.activate(self.set_state(x))


if __name__ == '__main__':
    n = Neuron(3)  # 3次元に設定
    x = np.array([1., 2., 0.])
    print(n.output(x))