import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import time
import sys
from scipy.special import erfc
from multiprocessing import Pool

N = 10**8
Ep = 1
Es_avg = 7
SNR_dB = np.arange(10, 22, 2)

CONSTELLATION = {
    1: (2, 2), # energy 8
    2: (2, -2), # energy 8
    3: (-2, -2), # energy 8
    4: (0, 2) # energy 4
}

def decide_symbol(point: Tuple[int, int]):
    if point[0] > 1 and point[1] > 0:
        return 1
    elif point[0] > 0 and point[1] < 0 and point[1] < 0.5 * (point[0] - 1):
        return 2
    elif point[0] < 0 and point[1] < -0.5 * (point[0] + 1):
        return 3
    else:
        return 4

def process_segment(args):
    start_ptr, amount, ep, n0, seed_offset = args
    np.random.seed(int(time.time()) + seed_offset)
    choices_segment = np.random.randint(1, 5, amount)
    symbols_segment = np.array([CONSTELLATION[r] for r in choices_segment])
    noise_1_segment = np.random.randn(amount) * np.sqrt(n0/2)
    noise_2_segment = np.random.randn(amount) * np.sqrt(n0/2)
    received_segment = symbols_segment + np.column_stack([noise_1_segment, noise_2_segment])
    decided_segment = np.array([decide_symbol(row) for row in received_segment])
    correct_segment = sum(decided_segment == choices_segment)

    return correct_segment


def main(argv):
    start = time.time()
    np.random.seed(int(time.time()))
    snr_linear = 10 ** (SNR_dB / 10)
    ser_results = []
    ber_theory = []
    upper_bounds = []
    lower_bounds = []

    num_processes = 10
    tex_data: Dict[int, List[str | float]] = {
        0: ['$N_0$'],
        1: ['$\\frac{E_p}{N_0}$'],
        2: ['$\\frac{E_p}{N_0}$ in dB'],
        3: ['$P_e$ Upper Bound'],
        4: ['$P_e$ Simulation'],
        5: ['$P_e$ Lower Bound'],
    }

    for snr in snr_linear:
        # SNR = Ep / (N0 / 2)
        # N0 = 2 * Ep / SNR
        n0 = Es_avg / snr
        tex_data[0].append(n0)

        # lets break it up so it doesn't eat all my RAM
        segment_amount = 10 ** 7
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
        ser_results.append((N - correct_total) / N)

        # inQ_val = np.sqrt(2 * Ep / n0)
        # ber_theory_val = 0.5 * erfc(inQ_val / np.sqrt(2))
        inQ = np.sqrt(2 * Es_avg / n0 / 7)
        upper_bound = 3 * erfc(inQ / np.sqrt(2))
        upper_bounds.append(upper_bound)
        lower_bound = 0.25 * erfc(inQ / np.sqrt(2))
        lower_bounds.append(lower_bound)
        # ber_theory.append(ber_theory_val)

        tex_data[1].append(Ep / n0)
        tex_data[2].append(10 * np.log10(Ep / n0))
        tex_data[3].append(upper_bound)
        tex_data[4].append((N - correct_total) / N)
        tex_data[5].append(lower_bound)

        print(f"N0: {n0:.8f} Ep/N0: {Ep / n0:.3f} Ep/N0 in DB: {10 * np.log10(Ep / n0):.3f} inQ: {inQ:.8f} SNR: {snr:.2f} CORRECT: {correct_total} SER sim: {ser_results[-1]:.5e} SER upper: {upper_bound:.5e} SER lower: {lower_bound:.5e}")
        print(f"ELAPSED: {time.time() - start:.2f}s")
    end = time.time()
    print("TOTAL TIME: ", end - start)

    print(ser_results)
    plot_data(ser_results, upper_bounds, lower_bounds)

    if '-t' in argv:
        print_tex_data(tex_data)

def print_tex_data(data):
    print(data)
    for k, vs in data.items():
        print(f"{k} & ", end="")
        for v in vs:
            print(f"{v:.3f} & ", end="")
        print("\\ ")

def Q(x, n):
    return 0.5 * erfc(x / np.sqrt(n)) 

def plot_data(data_sim, upper_bounds, lower_bounds):
    for s, u, l in zip(data_sim, upper_bounds, lower_bounds):
        print(f"SIM: {s:.4e}   UPPER: {u:.4e}   LOWER: {l:.4e}")
    plt.plot(SNR_dB, upper_bounds, c="#565149", label="Upper")
    plt.plot(SNR_dB, data_sim, 'o', c="#492365", label="Simulated")
    plt.plot(SNR_dB, lower_bounds, c="#565149", label="Lower")
    plt.legend()
    plt.title("Bit Error Rates for BAM")
    plt.xlabel("Signal-to-Noise Ratio (dB)")
    plt.ylabel("Error Probability")
    plt.yscale('log')
    plt.grid()
    plt.show()


def just_data():
    # calculated by an earlier run and preserved for the sake of convenience
    d_sim = [0.078652517, 0.037506918, 0.012500043, 0.002390372, 0.00019147, 3.879e-06, 4e-09]
    d_theory = [0.07864960352514258, 0.03750612835892601, 0.012500818040737566, 0.002388290780932807, 0.00019090777407599314, 3.87210821552205e-06, 9.006010350628787e-09]


    plot_data(d_sim, d_theory)


if __name__ == '__main__':
    if '-d' in sys.argv:
        just_data()
        exit()
    main(sys.argv)

