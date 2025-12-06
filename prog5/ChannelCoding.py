import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from pprint import pprint
import time
import sys
from scipy.special import erfc
from multiprocessing import Pool, cpu_count
import ctypes

channel_coding = ctypes.CDLL('./channel_coding.so')
channel_coding.hamming_errors.argtypes = [ctypes.c_longlong, ctypes.c_double, ctypes.c_int]
channel_coding.hamming_errors.restype = ctypes.c_int
channel_coding.uncoded_errors.argtypes = [ctypes.c_longlong, ctypes.c_double, ctypes.c_int]
channel_coding.uncoded_errors.restype = ctypes.c_int
channel_coding.convolution_errors.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_double, ctypes.c_int]
channel_coding.convolution_errors.restype = ctypes.c_int


start = time.time()

Ep = 1
# this dictionary shows the number of bits which will be processed for a given SNR in dB
snr_counts = {
    0: 10**7,
    1: 10**7,
    2: 10**7,
    3: 10**7,
    4: 10**7,       # e.g. 10**7 bits at 4 dB
    5: 10**8,
    6: 10**8,
    7: 10**8,
    8: 10**8,
    9: 10**9,
    10: 10**9,
    11: 10**9,
    12: 4 * 10**10,
}
SNR_dBs = np.array([k for k in snr_counts.keys()]) # Eb / N0 in dB
Ns = np.array([v for v in snr_counts.values()])

def Q(x, n=2):
    return 0.5 * erfc(x / np.sqrt(n))

def hamming_segment(args):
    amount, n0, seed_offset = args
    result = channel_coding.hamming_errors(amount, n0, seed_offset)
    print("HAM:", amount, round(n0, 2), result)
    return result

def uncoded_segment(args):
    amount, n0, seed_offset = args
    result = channel_coding.uncoded_errors(amount, n0, seed_offset)
    print("UNC:", amount, round(n0, 2), result)
    return result

def conv_segment(args):
    amount, ngroup, n0, seed_offset = args
    result = channel_coding.convolution_errors(amount, ngroup, n0, seed_offset)
    print("CON:", amount, round(n0, 2), result)
    return result

def format_time(sec):
    t = divmod(sec, 60)
    s = str(round(t[1], 2)).split('.')
    return f"{int(t[0])}:{s[0]:0>2}.{s[1]}"


def main():
    channel_coding.init_hash()

    num_processes = min(16, cpu_count())
    # segment_amount = 10**8 // 2
    ser_results = {
        'uncoded_theory': [],
        'uncoded_sim': [],

        'hamming_theory': [],
        'hamming_sim': [],

        'conv_theory': [],
        'conv_sim': [],
    }
    for snr_db, N in zip(SNR_dBs, Ns):
        print(f"Starting for {snr_db} dB.\t{format_time(time.time() - start)}")
        snr = 10 ** (snr_db / 10)
        segment_amount = N // num_processes
        assert segment_amount % 4 == 0
        n0 = Ep / snr
        ptr = 0
        uncoded_segments = []
        hamming_segments = []
        conv_segments = []
        seed_offset = 0

        # build the segments stuff
        while ptr < N:
            amount = min(segment_amount, N-ptr)
            uncoded_segments.append((amount, n0, seed_offset))
            hamming_segments.append((amount, n0 * 7 / 4, seed_offset))
            conv_segments.append((amount, 128, n0 * 2 / 1, seed_offset + 10000))
            ptr += segment_amount
            seed_offset += 1


        # uncoded
        # with Pool(processes=num_processes) as pool:
        #     uncoded_errors = pool.map(uncoded_segment, uncoded_segments)
        # uncoded_errors_total = sum(uncoded_errors)
        # ser_results['uncoded_sim'].append(uncoded_errors_total / N)

        ser_results['uncoded_theory'].append(Q(np.sqrt(2 * snr)))

        # hamming
        start_hamming = time.time()
        with Pool(processes=num_processes) as pool:
            hamming_errors = pool.map(hamming_segment, hamming_segments)
        hamming_errors_total = sum(hamming_errors)
        ser_results['hamming_sim'].append(hamming_errors_total / N)
        elapsed_hamming = time.time() - start_hamming
        print(f"Finished hamming at {snr_db} dB in {format_time(elapsed_hamming)}")

        # n0_hamming = n0
        # ber_hamming_theory = Q(np.sqrt(2 * Ep / n0_hamming))
        # ser_results['hamming_theory'].append(ber_hamming_theory)

        # convolutional
        start_conv = time.time()
        with Pool(processes=num_processes) as pool:
            conv_errors = pool.map(conv_segment, conv_segments)
        conv_errors_total = sum(conv_errors)
        ser_results['conv_sim'].append(conv_errors_total / N)
        elapsed_conv = time.time() - start_conv
        print(f"Finished convolutional at {snr_db} dB in {format_time(elapsed_conv)}")



    print("FINISHED in", format_time(time.time() - start))
    print_data(SNR_dBs, ser_results)
    plot_data(SNR_dBs, ser_results)

def plot_data(x, results):
    names = {
        'uncoded_theory': 'Theoretical',
        'hamming_sim': '(4, 7) Hamming',
        'conv_sim': '(2, 1, 2) Convolutional',
    }
    for name, vals in results.items():
        if len(x) != len(vals):
            continue
        print("PLOTTING:", name)
        plt.plot(x, vals, label=names[name] if name in names.keys() else name)
        plt.title("BER for Different Channel Coding Schemes")
        plt.xlabel("SNR in dB")
        plt.ylabel("BER")
        plt.legend()
        plt.semilogy()
    plt.show()

def print_data(x, results):
    data = {'SNR_dB': list(x), **results}
    pprint(data)
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
    if '-d' in sys.argv:
        data = {'SNR_dB': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                'conv_sim': [0.1934025,
                             0.1276522,
                             0.0707953,
                             0.0321059,
                             0.0115162,
                             0.00315798,
                             0.00066659,
                             0.0001038,
                             1.0874e-05,
                             7.85e-07,
                             3.1e-08,
                             1e-09,
                             1e-10],
                'conv_theory': [],
                'hamming_sim': [0.1189057,
                                0.0844746,
                                0.0549979,
                                0.0319137,
                                0.0161461,
                                0.00685079,
                                0.00233115,
                                0.00061387,
                                0.000116966,
                                1.5157e-05,
                                1.169e-06,
                                4.9e-08,
                                1.6e-09],
                'hamming_theory': [],
                'uncoded_sim': [],
                'uncoded_theory': [0.07864960352514258,
                                   0.05628195197654147,
                                   0.03750612835892598,
                                   0.022878407561085334,
                                   0.012500818040737566,
                                   0.005953867147778662,
                                   0.002388290780932807,
                                   0.0007726748153784446,
                                   0.00019090777407599314,
                                   3.36272284196176e-05,
                                   3.87210821552205e-06,
                                   2.613067953575205e-07,
                                   9.006010350628787e-09]}
        x = data['SNR_dB']
        results = {k: v for k, v in data.items() if k != 'SNR_dB'}
        plot_data(x, results)
        exit()
    main()
