import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from scipy.special import erfc
from multiprocessing import Pool, cpu_count

N = 10**4
Ep = 1
SNR_dB = np.array([0, 2, 4, 6, 8, 10, 12])

def process_segment(args):
    start_ptr, amount, ep, n0, seed_offset = args
    np.random.seed(int(time.time()) + seed_offset)
    bits_segment = np.random.randint(0, 2, amount)
    noise_segment = np.random.normal(0, np.sqrt(n0/2), amount)
    correct_segment = sum(((((bits_segment * 2 - 1) * ep) + noise_segment > 0).astype(int) == bits_segment).astype(int))

    return correct_segment


def main():
    start = time.time()
    np.random.seed(int(time.time()))
    snr_linear = 10 ** (SNR_dB / 10)
    ber_results = []
    ber_theory = []

    num_processes = 10

    for snr in snr_linear:
        # SNR = Ep / (N0 / 2)
        # N0 = 2 * Ep / SNR
        n0 = 2 * Ep / snr

        # lets break it up so it doesn't eat all my RAM
        segment_amount = 10 ** 8
        segments = []
        seed_offset = 0
        ptr = 0
        correct_total = 0

        while ptr < N:
            amount = min(segment_amount, N-ptr)
            segments.append((ptr, amount, Ep, n0, seed_offset))
            ptr += segment_amount
            seed_offset += 1


        with Pool(processes=num_processes) as pool:
            results = pool.map(process_segment, segments)

        correct_total = sum(results)
        ber_results.append((N - correct_total) / N)
        inQ_val = np.sqrt(2 * Ep / n0)
        ber_theory_val = 0.5 * erfc(inQ_val / np.sqrt(2))
        ber_theory.append(ber_theory_val)

        print(f"N0: {n0:.8f} Ep/N0: {Ep / n0:.3f} Ep/N0 in DB: {10 * np.log10(Ep / n0):.3f} inQ: {inQ_val:.8f} SNR: {snr:.2f} CORRECT: {correct_total} BER theory: {ber_theory_val:.5e} BER sim: {ber_results[-1]:.5e}")
        print(f"ELAPSED: {time.time() - start:.2f}s")
    end = time.time()
    print("TOTAL TIME: ", end - start)

    print(ber_results)
    plot_data(ber_results, ber_theory)

def Q(x, n):
    return 0.5 * erfc(x / np.sqrt(n)) 

def plot_data(data_sim, data_theory):
    for s, t in zip(data_sim, data_theory):
        print(f"SIM: {s:.4e}   THEORY: {t:.4e}")
    plt.plot(SNR_dB, data_theory, c="#565149", label="Theoretical")
    plt.plot(SNR_dB, data_sim, 'o', c="#492365", label="Simulated")
    plt.legend()
    plt.title("Bit Error Rates for BAM")
    plt.xlabel("Signal-to-Noise Ratio (dB)")
    plt.ylabel("Error Probability")
    plt.yscale('log')
    plt.grid()
    plt.show()


def just_data():
    # calculated by an earlier run and preserved for the sake of convenience
    d_sim = [0.158664578, 0.104024821, 0.056507059, 0.02300095, 0.006004842, 0.000781807, 3.4137e-05]
    d_theory = [ 1.58655e-01, 1.04029e-01, 5.64953e-02, 2.30071e-02, 6.00439e-03, 7.82701e-04, 3.43026e-05 ]


    plot_data(d_sim, d_theory)


if __name__ == '__main__':
    if '-d' in sys.argv:
        just_data()
        exit()
    main()

