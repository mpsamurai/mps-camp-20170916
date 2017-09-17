import numpy as np

class Layer:
    def __init__(self, m, n):
        self.W = np.random.rand(m,
                                n) - 0.5  # m行n列を設定  numpyのbroadcastで全ての行列に合わせて -0.5をしてくれるので 0 ~ 1の数を -0.5 ~ 0.5の範囲になる
        #         self.b = np.random.rand(m, 1)  # [] m行1列の要素をバイアスに与える
        self.b = np.zeros((m, 1))  # m行1列の　0ベクトルを作成してくれている  http://www.kamishima.net/mlmpyja/nbayes1/ndarray.html

    def set_state(self, x):
        s = (self.W.dot(x)) + 1 * self.b
        return s

    def activate(self, s):
        y = 1 / (1 + np.exp(-s))
        return y

    def output(self, x):
        return self.activate(self.set_state(x))


if __name__ == '__main__':
    n1 = Layer(2, 2)  # 設定
    x1 = np.array([[1.],
                   [2.]])
    print("1層目：", n1.output(x1))

    n2 = Layer(2, 2)
    x2 = n1.output(x1)
    print("2層目：", n2.output(x2))

    n3 = Layer(2, 2)
    x3 = n1.output(x2)
    print("2層目：", n3.output(x3))
