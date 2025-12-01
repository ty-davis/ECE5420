import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
import sys
from scipy.special import erfc
from multiprocessing import Pool, cpu_count

start = time.time()

N = 10**9
Ep = 1
SNR_dB = np.array([6, 8, 10, 12]) # Eb / N0 in dB

# SOME global matrices for the calculations
# hamming parity-check matrix
H = np.array([[0, 1, 1, 1, 1, 0, 0],
              [1, 0, 1, 1, 0, 1, 0],
              [1, 1, 0, 1, 0, 0, 1]])
# generator matrix
G = np.array([[1, 0, 0, 0, 0, 1, 1],
              [0, 1, 0, 0, 1, 0, 1],
              [0, 0, 1, 0, 1, 1, 0],
              [0, 0, 0, 1, 1, 1, 1]])

# errors matrix
E = np.array([list(reversed([1 if i == j else 0 for i in range(H.shape[1])])) for j in range(H.shape[1] + 1)])
# syndrome matrix
S = E @ H.T % 2
# syndrome translation dict
def hash(arr):
    total = 0
    l = len(arr)
    for i in range(l):
        total += arr[i] << (l-i - 1)
    return total

syn_tran = {hash(S[i]): E[i] for i in range(S.shape[0])}

# info and code matrices
info_mat = np.array([[int(x) for x in list(f"{bin(i)[2:]:0>4}")] for i in range(16)])
codes_mat = info_mat @ G % 2
codes_to_info = {hash(codes_mat[i]): info_mat[i] for i in range(codes_mat.shape[0])}

def Q(x, n=2):
    return 0.5 * erfc(x / np.sqrt(n))

def process_segment(args):
    amount, n0, seed_offset = args
    np.random.seed(int(time.time()) + seed_offset)
    data = np.random.randint(0, 16, amount // 4)
    codewords = np.array([codes_mat[x] for x in data])
    codewords_norm = (codewords * 2 - 1) * Ep
    print(f"1: {time.time() - start:.1f}", seed_offset)

    noise = np.random.randn(*codewords.shape) * np.sqrt(n0/2)
    received = (codewords_norm + noise > 0).astype(int)

    del codewords, noise
    print(f"2: {time.time() - start:.1f}", seed_offset)

    syndromes = np.array([hash(row) for row in received @ H.T % 2])
    error_patterns = [syn_tran[h] for h in syndromes]
    received ^= error_patterns
    print(f"3: {time.time() - start:.1f}", seed_offset)
    del syndromes, error_patterns

    info_received = np.array([codes_to_info[hash(row)] for row in received])
    data_bits = np.array([info_mat[d] for d in data])

    errors = np.sum(data_bits != info_received)
    print(f"4: {time.time() - start:.1f}", seed_offset)
    return errors


def main():
    print(E, S, sep="\n")
    snr_linear = 10 ** (SNR_dB / 10)

    num_processes = 3
    segment_amount = 10**8
    assert segment_amount % 4 == 0
    ser_results = []
    for snr in snr_linear:
        print(f"{snr}", time.time())
        n0 = Ep / snr
        ptr = 0
        segments = []
        seed_offset = 0

        while ptr < N:
            amount = min(segment_amount, N-ptr)
            segments.append((amount, n0, seed_offset))
            ptr += segment_amount
            seed_offset += 1

        with Pool(processes=num_processes) as pool:
            results = pool.map(process_segment, segments)

        errors_total = sum(results)
        ser_results.append(errors_total / N)
        print(errors_total / N, "\n\n")

    plot_data(SNR_dB, ser_results)

def plot_data(x, y):
    plt.plot(x, y)
    plt.show()


# if __name__ == '__main__':
#     main()


# notes
# Need to take 10**9 bits and split them into groups of 4
# groups of 4 are translated into codewords (7 bits long)
# simulate transmission over gaussian channel and correct the bits
# gather data at the end and compare results
