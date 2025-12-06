import ctypes

channel_coding = ctypes.CDLL('./channel_coding.so')
channel_coding.convolution_errors.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int]
channel_coding.convolution_errors.restype = ctypes.c_int

print("ERRORS:", channel_coding.convolution_errors(100, 5, 3, 0))
