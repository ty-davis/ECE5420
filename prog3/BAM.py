import numpy as np
import matplotlib.pyplot as plt
import time

N = 10**9
Ep = 1
SNR_dB = np.array([0, 2, 4, 6, 8, 10, 12])

def main():
    start = time.time()
    np.random.seed(int(time.time()))
    snr_linear = 10 ** (SNR_dB / 10)
    ber_results = []
    for snr in snr_linear:
        # SNR = Ep / (N0 / 2)
        # N0 = 2 * Ep / SNR
        n0 = 2 * Ep / snr
        # bits = np.random.randint(0, 2, N)
        # noise = np.random.normal(0, np.sqrt(n0/2), N)
        # lets break it up so it doesn't eat all my RAM

        segment_amount = 10 ** 8
        ptr = 0
        correct_total = 0
        while ptr <= N:
            amount = min(ptr, N)
            bits_segment = np.random.randint(0, 2, amount)
            noise_segment = np.random.normal(0, np.sqrt(n0/2), amount)
            # bits_segment = bits[ptr:ptr+segment_amount]
            # noise_segment = noise[ptr:ptr+segment_amount]
            correct_segment = sum(((((bits_segment * 2 - 1) * Ep) + noise_segment > 0).astype(int) == bits_segment).astype(int))
            correct_total += correct_segment
            print(ptr, correct_segment, "\tELAPSED:", time.time() - start)
            ptr += segment_amount
        # mapped_bits = (bits * 2 - 1) * Ep
        # received = mapped_bits + noise
        # decided = (received > 0).astype(int)
        # correct_bits = (decided == bits).astype(int)
        # correct_total = sum(correct_bits)
        # correct_total = sum(((((bits * 2 - 1) * Ep) + noise > 0).astype(int) == bits).astype(int))
        ber_results.append((N - correct_total) / N)

        print(n0)
        # print(bits)
        # print(mapped_bits)
        # print(received)
        # print(decided)
        # print(correct_bits)
        print(correct_total)



        print("\n----\n")
    end = time.time()
    print("TOTAL TIME: ", end - start)

    print(ber_results)
    plt.plot(SNR_dB, ber_results)
    plt.yscale('log')
    plt.grid()
    plt.show()



if __name__ == '__main__':
    main()
