import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from pprint import pprint
import time
import sys
from scipy.special import erfc
from multiprocessing import Pool, cpu_count
import ctypes
import platform

lib_name = './channel_coding.dll' if platform.system() == 'Windows' else './channel_coding.so'
channel_coding = ctypes.CDLL(lib_name)
channel_coding.hamming_errors.argtypes = [ctypes.c_longlong, ctypes.c_double, ctypes.c_int]
channel_coding.hamming_errors.restype = ctypes.c_int
channel_coding.uncoded_errors.argtypes = [ctypes.c_longlong, ctypes.c_double, ctypes.c_int]
channel_coding.uncoded_errors.restype = ctypes.c_int
channel_coding.convolution_errors.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_double, ctypes.c_int]
channel_coding.convolution_errors.restype = ctypes.c_int


start = time.time()

# this dictionary shows the number of bits which will be processed for a given SNR in dB
snr_bitcounts = {
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
    11: 10**10,
    12: 4 * 10**10,
}
SNR_dBs = np.array([k for k in snr_bitcounts.keys()]) # Eb / N0 in dB
Ns = np.array([v for v in snr_bitcounts.values()])
Eb = 1

def Q(x, n=2):
    return 0.5 * erfc(x / np.sqrt(n))

def worker_init():
    channel_coding.init_hash()

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


def simulate_data():
    channel_coding.init_hash()

    num_processes = min(28, cpu_count())
    # segment_amount = 10**8 // 2
    ser_results = {
        'uncoded_theory': [],
        'uncoded_sim': [],

        'hamming_sim': [],

        'conv_sim': [],
    }
    for snr_db, N in zip(SNR_dBs, Ns):
        print(f"Starting for {snr_db} dB.\t{format_time(time.time() - start)}")
        snr = 10 ** (snr_db / 10)
        segment_amount = N // num_processes
        n0 = Eb / snr
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
        with Pool(processes=num_processes) as pool:
            uncoded_errors = pool.map(uncoded_segment, uncoded_segments)
        uncoded_errors_total = sum(uncoded_errors)
        ser_results['uncoded_sim'].append(uncoded_errors_total / N)

        ser_results['uncoded_theory'].append(Q(np.sqrt(2 * snr)))

        # hamming
        start_hamming = time.time()
        with Pool(processes=num_processes, initializer=worker_init) as pool:
            hamming_errors = pool.map(hamming_segment, hamming_segments)
        hamming_errors_total = sum(hamming_errors)
        ser_results['hamming_sim'].append(hamming_errors_total / N)
        elapsed_hamming = time.time() - start_hamming
        print(f"Finished hamming at {snr_db} dB in {format_time(elapsed_hamming)}")

        # convolutional
        start_conv = time.time()
        with Pool(processes=num_processes) as pool:
            conv_errors = pool.map(conv_segment, conv_segments)
        conv_errors_total = sum(conv_errors)
        ser_results['conv_sim'].append(conv_errors_total / N)
        elapsed_conv = time.time() - start_conv
        print(f"Finished convolutional at {snr_db} dB in {format_time(elapsed_conv)}")


    print("FINISHED SIMULATION in", format_time(time.time() - start))
    return SNR_dBs, ser_results

def round_sf(x, n):
    """Round x to n sig figs"""
    if x == 0:
        return 0
    return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))

def plot_data(x, results):
    names = {
        'uncoded_sim': 'Without Encoding',
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

def print_latex_table(x, results):
    snr_to_use = [6, 8, 10, 12]
    snr_linear = [10 ** (snr/10) for snr in snr_to_use]
    idxs = [list(x).index(i) for i, v in enumerate(x) if v in snr_to_use]
    # res = [results[i] for i in idxs]

    # Eb / n_0 = SNR
    Ebs = [1 for snr in snr_linear]
    Ec_ham = [eb * 4 / 7 for eb in Ebs]
    Ec_conv = [eb * 1 / 2 for eb in Ebs]
    N_0s = [1 / eb / snr for eb, snr in zip(Ebs, snr_linear)]
    N_0_ham = [1 / eb / snr for eb, snr in zip(Ec_ham, snr_linear)]
    N_0_conv = [1 / eb / snr for eb, snr in zip(Ec_conv, snr_linear)]
    # eb / n0 in dB
    P_b_in_q = [np.sqrt(2 * snr) for snr in snr_linear]
    P_b_theory = [Q(pb) for pb in P_b_in_q]
    P_b_sim = [results['uncoded_sim'][i] if i < len(results['uncoded_sim']) else 0 for i in idxs]
    P_b_sim_ham = [results['hamming_sim'][i] for i in idxs]
    P_b_sim_conv = [results['conv_sim'][i] for i in idxs]

    row_titles = [
        '$E_b$',
        '$E_c$ for (7, 4) Hamming Code',
        '$E_c$ for Convolutional Code',
        '$N_0$',
        '$N_0$ for (7, 4) Hamming Code',
        '$N_0$ for Convolutional Code',
        '$\\frac{E_b}{N_0}$ in dB',
        '$P_b$ in Q function',
        '$P_b$ theoretical value',
        '$P_b$ Simulated without Encoding',
        '$P_b$ Simulated with (7, 4) Hamming Code',
        '$P_b$ Simulated with Convolutional Code',
    ]
    for j, row in enumerate([Ebs,
                             Ec_ham,
                             Ec_conv,
                             N_0s,
                             N_0_ham,
                             N_0_conv,
                             snr_to_use,
                             P_b_in_q,
                             P_b_theory,
                             P_b_sim,
                             P_b_sim_ham,
                             P_b_sim_conv,
                             ]):
        print(row_titles[j], end=" & ")
        for i in range(4):
            if i < 3:
                print(round_sf(row[i], 3), end=' & ')
            else:
                print(round_sf(row[i], 3), end=' \\\\')
        print()



if __name__ == '__main__':
    if '-d' in sys.argv:
        data =  {'SNR_dB': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                 'conv_sim': [0.1933896,
                              0.127401,
                              0.0709712,
                              0.0320615,
                              0.0114453,
                              0.0031646,
                              0.00066271,
                              0.00010012,
                              1.107e-05,
                              8.46e-07,
                              4.6e-08,
                              1.6e-09,
                              2.5e-11],
                 'hamming_sim': [0.1189188,
                                 0.0846048,
                                 0.0549236,
                                 0.0318923,
                                 0.0163133,
                                 0.00687367,
                                 0.00233411,
                                 0.00061166,
                                 0.00010711,
                                 1.5434e-05,
                                 1.144e-06,
                                 4.91e-08,
                                 8.75e-10],
                 'uncoded_sim': [0.078737,
                                  0.0563434,
                                  0.0375078,
                                  0.0228655,
                                  0.0124696,
                                  0.00595641,
                                  0.00239317,
                                  0.00077609,
                                  0.00019333,
                                  3.43e-05,
                                  3.9e-06,
                                  2.9e-07,
                                  1.5e-08],
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
    else:
        x, results = simulate_data()
    print_latex_table(x, results)
    print_data(x, results)
    plot_data(x, results)
