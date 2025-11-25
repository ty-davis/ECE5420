import numpy as np
import matplotlib.pyplot as plt

class BMatrix:
    def __init__(self, data):
        self.data = data

    def _mod2(self, data):
        return data % 2

    def __add__(self, other):
        res = BMatrix(self.data + other.data)
        return self._mod2(res)

    def __mul__(self, scalar):
        return self._mod2(self.data * scalar)

    def __matmul__(self, other):
        return self._mod2(self.data @ other.data)

H = np.array([[0, 1, 1, 1, 1, 0, 0],
              [1, 0, 1, 1, 0, 1, 0],
              [1, 1, 0, 1, 0, 0, 1]])
G = np.array([[1, 0, 0, 0, 0, 1, 1],
              [0, 1, 0, 0, 1, 0, 1],
              [0, 0, 1, 0, 1, 1, 0],
              [0, 0, 0, 1, 1, 1, 1]])

E = np.array([list(reversed([1 if i == j else 0 for i in range(H.shape[1])])) for j in range(H.shape[1] + 1)])
S = E @ H.T


def main():
    print(H.shape)
    print(G.shape)
    print(E)
    print(S)


if __name__ == '__main__':
    main()
