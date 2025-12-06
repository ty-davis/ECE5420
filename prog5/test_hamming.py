import ctypes
import platform

lib_name = './channel_coding.dll' if platform.system() == 'Windows' else './channel_coding.so'
channel_coding = ctypes.CDLL(lib_name)
channel_coding.hamming_errors.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int]
channel_coding.hamming_errors.restype = ctypes.c_int
channel_coding.uncoded_errors.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int]
channel_coding.uncoded_errors.restype = ctypes.c_int

snr_db = 4
n0 = 1 / 10 ** (snr_db/10)
channel_coding.init_hash()
print(channel_coding.hamming_errors(100, n0, 0))
