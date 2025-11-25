import numpy as np
import matplotlib.pyplot as plt

class BMatrix(np.ndarray):
    def __new__(cls, data):
        obj = np.asarray(data).view(cls)
        return obj

    def _mod2(self, data):
        return data % 2

    def __add__(self, other):
        res = super().__add__(other)
        return self._mod2(res).view(BMatrix)

    def __mul__(self, scalar):
        result = super().__mul__(scalar)
        return self._mod2(result).view(BMatrix)

    def __matmul__(self, other):
        result = super().__matmul__(other)
        return self._mod2(result).view(BMatrix)

H = BMatrix([[0, 1, 1, 1, 1, 0, 0],
              [1, 0, 1, 1, 0, 1, 0],
              [1, 1, 0, 1, 0, 0, 1]])
G = BMatrix([[1, 0, 0, 0, 0, 1, 1],
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
    ones = np.ones(S.shape, dtype=int)
    print(S + ones)



if __name__ == '__main__':
    main()
