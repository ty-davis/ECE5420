import numpy as np
import matplotlib.pyplot as plt

def main():
    fig, axs = plt.subplots(2, 2,)

    t = np.arange(-10, 10, 0.01)
    dt = t[1] - t[0]
    y1 = np.sinc(t)
    y2 = y1 ** 2

    Y1 = np.fft.fft(y1) * dt
    Y2 = np.fft.fft(y2) * dt

    N = len(t)
    freqs = np.fft.fftfreq(N, dt)

    freqs_shifted = np.fft.fftshift(freqs)
    Y1_shifted = np.fft.fftshift(Y1)
    Y2_shifted = np.fft.fftshift(Y2)

    # time domain plots
    axs[0, 0].plot(t, y1)
    axs[0, 0].set_title('Sinc Function')
    axs[1, 0].plot(t, y2)
    axs[1, 0].set_title('Sinc$^2$ Function')

    # frequency domain plots
    axs[0, 1].plot(freqs_shifted, np.abs(Y1_shifted))
    axs[0, 1].set_title('FT of Sinc$(t)$')
    axs[0, 1].set_xlim(-5, 5)
    axs[1, 1].plot(freqs_shifted, np.abs(Y2_shifted))
    axs[1, 1].set_title('FT of Sinc$^2(t)$')
    axs[1, 1].set_xlim(-5, 5)

    plt.tight_layout()
    plt.show()
    plt.savefig('output.png')

if __name__ == '__main__':
    main()
