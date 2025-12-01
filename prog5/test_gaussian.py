import ctypes
import time

channel_coding = ctypes.CDLL('./channel_coding.so')
channel_coding.gaussian_noise.restype = ctypes.c_double
channel_coding.seed_xoro.argtypes = [ctypes.c_int]

N = 100000000
channel_coding.seed_xoro(int(time.time()))
x = [channel_coding.gaussian_noise() for _ in range(N)]

# grab the tails at each std dev to verify that it's a good gaussian distribution
for i in range(1, 5):
    print(f"{i} std dev from mean")
    left_tail = [n for n in x if n < -1 * i]
    right_tail = [n for n in x if n > 1 * i]
    left_len = len(left_tail)
    right_len = len(right_tail)
    print("LEFT:", left_len / N)
    print("RIGHT:", right_len / N)
    print("INSIDE:", 1 - (left_len + right_len) / N)
    print("OUTSIDE:", (left_len + right_len) / N)
    print()

# import matplotlib.pyplot as plt
# plt.hist(x, bins=500)
# plt.show()

