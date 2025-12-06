#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <limits.h>

#define INDEX(row, col, num_cols) ((row) * (num_cols) + (col))
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device functions (run on GPU)
__device__ int arrdiff_device(int* arr1, int* arr2, int size) {
    int total = 0;
    for (int i = 0; i < size; i++) {
        total += arr1[i] == arr2[i] ? 0 : 1;
    }
    return total;
}

__device__ void arrcpy_device(int* dest, int* tar, int len) {
    for (int i = 0; i < len; i++) {
        dest[i] = tar[i];
    }
}

__device__ float gaussian_noise_device(curandState* state) {
    return curand_normal(state);
}

// Convolutional coding lookup tables (constant memory for fast access)
__constant__ int d_state_outputs[8][2] = {
    {0, 0}, {1, 1}, {1, 1}, {0, 0},
    {1, 0}, {0, 1}, {0, 1}, {1, 0}
};

__constant__ int d_last_states[4][2] = {
    {0, 1}, {2, 3}, {0, 1}, {2, 3}
};

// Hamming code lookup tables
__constant__ int d_codewords[16][7] = {
    {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 1, 1, 1},
    {0, 0, 1, 0, 1, 1, 0}, {0, 0, 1, 1, 0, 0, 1},
    {0, 1, 0, 0, 1, 0, 1}, {0, 1, 0, 1, 0, 1, 0},
    {0, 1, 1, 0, 0, 1, 1}, {0, 1, 1, 1, 1, 0, 0},
    {1, 0, 0, 0, 0, 1, 1}, {1, 0, 0, 1, 1, 0, 0},
    {1, 0, 1, 0, 1, 0, 1}, {1, 0, 1, 1, 0, 1, 0},
    {1, 1, 0, 0, 1, 1, 0}, {1, 1, 0, 1, 0, 0, 1},
    {1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 1}
};

__constant__ int d_h_t[7][3] = {
    {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
    {1, 0, 0}, {0, 1, 0}, {0, 0, 1}
};

typedef struct {
    int bit;
    int num_errors;
    int last_state;
} node;

// Uncoded kernel
__global__ void uncoded_kernel(int* errors, double n_0, int n_per_thread, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    int local_errors = 0;
    
    for (int i = 0; i < n_per_thread; i++) {
        int bit = curand(&state) & 1;
        double received = (double)bit * 2.0 - 1.0;
        received += gaussian_noise_device(&state) * sqrt(n_0 / 2.0);
        local_errors += (received > 0 && bit) || (received <= 0 && !bit) ? 0 : 1;
    }
    
    atomicAdd(errors, local_errors);
}

// Hamming kernel helper functions
__device__ int hash_device(int* arr, int arr_length) {
    int total = 0;
    for (int i = 0; i < arr_length; i++) {
        total += arr[i] << (arr_length - i - 1);
    }
    return total;
}

__device__ int error_idx_device(int* arr) {
    int s[3] = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 7; j++) {
            s[i] += arr[j] * d_h_t[j][i];
        }
        s[i] = s[i] % 2;
    }
    
    switch (hash_device(s, 3)) {
        case 1: return 6;
        case 2: return 5;
        case 4: return 4;
        case 7: return 3;
        case 6: return 2;
        case 5: return 1;
        case 3: return 0;
        default: return -1;
    }
}

__device__ void data_to_bits_device(int* dest, int n, int size) {
    for (int i = size - 1; i >= 0; i--) {
        dest[i] = n % 2;
        n /= 2;
    }
}

// Hamming kernel
__global__ void hamming_kernel(int* errors, double n_0, int n_groups_per_thread, 
                               unsigned long long seed, int* hash_to_bucket) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    int local_errors = 0;
    
    for (int group = 0; group < n_groups_per_thread; group++) {
        int data = curand(&state) % 16;
        int bits[4];
        data_to_bits_device(bits, data, 4);
        
        int codeword[7];
        int hash_val = hash_device(bits, 4);
        for (int j = 0; j < 7; j++) {
            codeword[j] = d_codewords[hash_val][j];
        }
        
        double received[7];
        for (int j = 0; j < 7; j++) {
            received[j] = (double)(codeword[j] * 2 - 1);
            received[j] += gaussian_noise_device(&state) * sqrt(n_0 / 2.0);
        }
        
        int codeword_received[7];
        for (int j = 0; j < 7; j++) {
            codeword_received[j] = received[j] > 0 ? 1 : 0;
        }
        
        int error = error_idx_device(codeword_received);
        if (error >= 0) {
            codeword_received[error] = (codeword_received[error] + 1) % 2;
        }
        
        int data_received = hash_to_bucket[hash_device(codeword_received, 7)];
        int bits_received[4];
        data_to_bits_device(bits_received, data_received, 4);
        
        local_errors += arrdiff_device(bits, bits_received, 4);
    }
    
    atomicAdd(errors, local_errors);
}

// Convolutional kernel with reduced memory footprint
__global__ void convolution_kernel(int* errors, int ngroup, double n_0, 
                                   int n_messages_per_thread, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    int total_length = ngroup + 2;
    int local_errors = 0;
    
    // Limit to reasonable size for local memory
    if (total_length > 130) {
        return;
    }
    
    for (int msg = 0; msg < n_messages_per_thread; msg++) {
        int bits[132];  // ngroup + 2, max 130
        int received[264];  // (ngroup + 2) * 2
        int decoded[132];
        
        int memory[2] = {0, 0};
        
        // Encoding
        for (int i = 0; i < total_length; i++) {
            if (i < ngroup) {
                bits[i] = curand(&state) & 1;
            } else {
                bits[i] = 0;
            }
            
            double coded_1 = (double)((bits[i] ^ memory[0] ^ memory[1]) * 2 - 1);
            double coded_2 = (double)((bits[i] ^ memory[1]) * 2 - 1);
            coded_1 += gaussian_noise_device(&state) * sqrt(n_0 / 2.0);
            coded_2 += gaussian_noise_device(&state) * sqrt(n_0 / 2.0);
            
            received[INDEX(i, 0, 2)] = coded_1 > 0 ? 1 : 0;
            received[INDEX(i, 1, 2)] = coded_2 > 0 ? 1 : 0;
            
            memory[1] = memory[0];
            memory[0] = bits[i];
        }
        
        // Viterbi decoding - process one time step at a time to reduce memory
        // Use two arrays for current and previous trellis states
        node prev_trellis[4];
        node curr_trellis[4];
        
        // Initialize
        for (int i = 0; i < 4; i++) {
            prev_trellis[i].num_errors = INT_MAX / 2;
        }
        prev_trellis[0].num_errors = 0;
        prev_trellis[0].bit = 0;
        
        int decode_bits[2];
        decode_bits[0] = received[0];
        decode_bits[1] = received[1];
        
        prev_trellis[0].num_errors = arrdiff_device(decode_bits, (int*)d_state_outputs[0], 2);
        prev_trellis[0].bit = 0;
        prev_trellis[0].last_state = 0;
        prev_trellis[2].num_errors = arrdiff_device(decode_bits, (int*)d_state_outputs[1], 2);
        prev_trellis[2].bit = 1;
        prev_trellis[2].last_state = 0;
        
        // Store path history in a compact format
        char path_history[132][4];  // [time][state] -> last_state
        path_history[0][0] = 0;
        path_history[0][2] = 0;
        
        for (int i = 1; i < total_length; i++) {
            decode_bits[0] = received[INDEX(i, 0, 2)];
            decode_bits[1] = received[INDEX(i, 1, 2)];
            
            for (int state = 0; state < 4; state++) {
                curr_trellis[state].num_errors = INT_MAX;
                for (int k = 0; k < 2; k++) {
                    int last_state = d_last_states[state][k];
                    int diff_errors = arrdiff_device(decode_bits, 
                        (int*)d_state_outputs[INDEX(last_state, state / 2, 2)], 2);
                    int last_errors = prev_trellis[last_state].num_errors;
                    int test_errors = diff_errors + last_errors;
                    
                    if (test_errors < curr_trellis[state].num_errors) {
                        curr_trellis[state].num_errors = test_errors;
                        curr_trellis[state].bit = state / 2;
                        curr_trellis[state].last_state = last_state;
                        path_history[i][state] = last_state;
                    }
                }
            }
            
            // Copy current to previous for next iteration
            for (int s = 0; s < 4; s++) {
                prev_trellis[s] = curr_trellis[s];
            }
        }
        
        // Traceback
        int this_state = 0;
        int fewest_errors = INT_MAX;
        for (int state = 0; state < 4; state++) {
            int num_errors = curr_trellis[state].num_errors;
            if (num_errors < fewest_errors) {
                fewest_errors = num_errors;
                this_state = state;
            }
        }
        
        for (int i = total_length - 1; i >= 0; i--) {
            decoded[i] = this_state / 2;
            this_state = path_history[i][this_state];
        }
        
        local_errors += arrdiff_device(decoded, bits, ngroup);
    }
    
    atomicAdd(errors, local_errors);
}

// Host wrapper functions
extern "C" {
    int uncoded_errors(long long ndata, double n_0, int seed_offset) {
        int* d_errors;
        int h_errors = 0;
        
        CUDA_CHECK(cudaMalloc(&d_errors, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_errors, &h_errors, sizeof(int), cudaMemcpyHostToDevice));
        
        int threads_per_block = 256;
        int num_blocks = 256;
        int total_threads = threads_per_block * num_blocks;
        int n_per_thread = (ndata + total_threads - 1) / total_threads;
        
        unsigned long long seed = time(NULL) + seed_offset;
        
        uncoded_kernel<<<num_blocks, threads_per_block>>>(d_errors, n_0, n_per_thread, seed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_errors));
        
        return h_errors;
    }
    
    void init_hash() {
        // Hash table initialization on CPU
        static int hash_to_bucket[257];
        for (int i = 0; i < 257; i++) hash_to_bucket[i] = -1;
        
        int codewords_cpu[16][7] = {
            {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 1, 1, 1},
            {0, 0, 1, 0, 1, 1, 0}, {0, 0, 1, 1, 0, 0, 1},
            {0, 1, 0, 0, 1, 0, 1}, {0, 1, 0, 1, 0, 1, 0},
            {0, 1, 1, 0, 0, 1, 1}, {0, 1, 1, 1, 1, 0, 0},
            {1, 0, 0, 0, 0, 1, 1}, {1, 0, 0, 1, 1, 0, 0},
            {1, 0, 1, 0, 1, 0, 1}, {1, 0, 1, 1, 0, 1, 0},
            {1, 1, 0, 0, 1, 1, 0}, {1, 1, 0, 1, 0, 0, 1},
            {1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 1}
        };
        
        for (int i = 0; i < 16; i++) {
            int hash_val = 0;
            for (int j = 0; j < 7; j++) {
                hash_val += codewords_cpu[i][j] << (6 - j);
            }
            hash_to_bucket[hash_val] = i;
        }
    }
    
    int hamming_errors(long long ndata, double n_0, int seed_offset) {
        static int* d_hash_to_bucket = nullptr;
        if (d_hash_to_bucket == nullptr) {
            int hash_to_bucket[257];
            for (int i = 0; i < 257; i++) hash_to_bucket[i] = -1;
            
            int codewords_cpu[16][7] = {
                {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 1, 1, 1},
                {0, 0, 1, 0, 1, 1, 0}, {0, 0, 1, 1, 0, 0, 1},
                {0, 1, 0, 0, 1, 0, 1}, {0, 1, 0, 1, 0, 1, 0},
                {0, 1, 1, 0, 0, 1, 1}, {0, 1, 1, 1, 1, 0, 0},
                {1, 0, 0, 0, 0, 1, 1}, {1, 0, 0, 1, 1, 0, 0},
                {1, 0, 1, 0, 1, 0, 1}, {1, 0, 1, 1, 0, 1, 0},
                {1, 1, 0, 0, 1, 1, 0}, {1, 1, 0, 1, 0, 0, 1},
                {1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 1}
            };
            
            for (int i = 0; i < 16; i++) {
                int hash_val = 0;
                for (int j = 0; j < 7; j++) {
                    hash_val += codewords_cpu[i][j] << (6 - j);
                }
                hash_to_bucket[hash_val] = i;
            }
            
            CUDA_CHECK(cudaMalloc(&d_hash_to_bucket, 257 * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_hash_to_bucket, hash_to_bucket, 257 * sizeof(int), 
                                  cudaMemcpyHostToDevice));
        }
        
        int* d_errors;
        int h_errors = 0;
        
        CUDA_CHECK(cudaMalloc(&d_errors, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_errors, &h_errors, sizeof(int), cudaMemcpyHostToDevice));
        
        int threads_per_block = 256;
        int num_blocks = 256;
        int total_threads = threads_per_block * num_blocks;
        int n_groups = ndata / 4;
        int n_groups_per_thread = (n_groups + total_threads - 1) / total_threads;
        
        unsigned long long seed = time(NULL) + seed_offset;
        
        hamming_kernel<<<num_blocks, threads_per_block>>>(d_errors, n_0, n_groups_per_thread, 
                                                           seed, d_hash_to_bucket);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_errors));
        
        return h_errors;
    }
    
    int convolution_errors(long long ndata, int ngroup, double n_0, int seed_offset) {
        int* d_errors;
        int h_errors = 0;
        
        CUDA_CHECK(cudaMalloc(&d_errors, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_errors, &h_errors, sizeof(int), cudaMemcpyHostToDevice));
        
        int threads_per_block = 64; // Reduced from 128 to give more memory per thread
        int num_blocks = 256; // Increased blocks to compensate
        int total_threads = threads_per_block * num_blocks;
        int n_messages = ndata / ngroup;
        int n_messages_per_thread = (n_messages + total_threads - 1) / total_threads;
        
        unsigned long long seed = time(NULL) + seed_offset;
        
        convolution_kernel<<<num_blocks, threads_per_block>>>(d_errors, ngroup, n_0, 
                                                               n_messages_per_thread, seed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_errors));
        
        return h_errors;
    }
}
