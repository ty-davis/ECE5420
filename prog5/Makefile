NVCC=nvcc
CUDA_FLAGS=-arch=sm_89 -Xcompiler -fPIC --shared
CUDA_LIBS=-lcurand

all: channel_coding.so

channel_coding.so: channel_coding.cu
	$(NVCC) $(CUDA_FLAGS) channel_coding.cu -o channel_coding.so $(CUDA_LIBS)

clean:
	rm -f channel_coding.so

.PHONY: all clean
