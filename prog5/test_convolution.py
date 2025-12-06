import ctypes
import platform

lib_name = './channel_coding.dll' if platform.system() == 'Windows' else './channel_coding.so'
channel_coding = ctypes.CDLL(lib_name)

channel_coding.convolution_errors.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int]
channel_coding.convolution_errors.restype = ctypes.c_int

print("ERRORS:", channel_coding.convolution_errors(100, 5, 3, 0))
