import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
import sys
from scipy.special import erfc
from multiprocessing import Pool, cpu_count
import ctypes

channel_coding = ctypes.CDLL('./channel_coding.so')
channel_coding.hamming_errors.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int]
channel_coding.hamming_errors.restype = ctypes.c_int
channel_coding.uncoded_errors.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int]
channel_coding.uncoded_errors.restype = ctypes.c_int


start = time.time()

N = 10**9
Ep = 1
SNR_dB = np.array([6, 8, 10, 12]) # Eb / N0 in dB

def Q(x, n=2):
    return 0.5 * erfc(x / np.sqrt(n))

def hamming_segment(args):
    amount, n0, seed_offset = args
    result = channel_coding.hamming_errors(amount, n0, seed_offset)
    print(amount, n0, result)
    return result

def uncoded_segment(args):
    amount, n0, seed_offset = args
    result = channel_coding.uncoded_errors(amount, n0, seed_offset)
    print(amount, n0, result)
    return result

def main():
    channel_coding.init_hash()
    snr_linear = 10 ** (SNR_dB / 10)

    num_processes = 16
    segment_amount = 10**8 // 2
    assert segment_amount % 4 == 0
    ser_results = {
        'uncoded_theory': [],
        'uncoded_sim': [],

        'hamming_theory': [],
        'hamming_sim': [],

        'conv_theory': [],
        'conv_sim': [],
    }
    for snr in snr_linear:
        print(f"{snr}", time.time() - start)
        n0 = Ep / snr
        ptr = 0
        uncoded_segments = []
        hamming_segments = []
        conv_segments = []
        seed_offset = 0

        # build the segments stuff
        while ptr < N:
            amount = min(segment_amount, N-ptr)
            hamming_segments.append((amount, n0, seed_offset))
            uncoded_segments.append((amount, n0, seed_offset + 100))
            conv_segments.append((amount, n0, seed_offset + 10000))
            ptr += segment_amount
            seed_offset += 1


        # uncoded
        with Pool(processes=num_processes) as pool:
            uncoded_errors = pool.map(uncoded_segment, uncoded_segments)
        uncoded_errors_total = sum(uncoded_errors)
        ser_results['uncoded_sim'].append(uncoded_errors_total / N)

        ser_results['uncoded_theory'].append(Q(np.sqrt(2 * snr)))

        # hamming
        with Pool(processes=num_processes) as pool:
            hamming_errors = pool.map(hamming_segment, hamming_segments)
        hamming_errors_total = sum(hamming_errors)
        ser_results['hamming_sim'].append(hamming_errors_total / N)

        n0_hamming = n0 * 4 / 7
        ber_hamming_theory = Q(np.sqrt(2 * Ep / n0_hamming))
        ser_results['hamming_theory'].append(ber_hamming_theory)

        # convolutional




    print("FINISHED", time.time() - start)
    print_data(SNR_dB, ser_results)
    plot_data(SNR_dB, ser_results)

def plot_data(x, results):
    for name, vals in results.items():
        if len(x) != len(vals):
            continue
        print("PLOTTING:", name)
        plt.plot(x, vals, label=name)
        plt.legend()
        plt.semilogy()
    plt.show()

def print_data(x, results):
    keys = [key for key in results.keys() if len(x) == len(results[key])]
    print(f"SNR_dB", end="\t")
    for key in keys:
        print(key, end="\t")
    print()
    for i in range(len(x)):
        print(f"{x[i]}", end='\t')
        for key in keys:
            print(f"{results[key][i]:0.5e}", end='\t')

        print()




if __name__ == '__main__':
    main()


# notes
# Need to take 10**9 bits and split them into groups of 4
# groups of 4 are translated into codewords (7 bits long)
# simulate transmission over gaussian channel and correct the bits
# gather data at the end and compare results
