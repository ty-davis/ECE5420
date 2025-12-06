#include <stdio.h>
#include <stdlib.h>
#include "noise.h"
#include <time.h>
#include <math.h>
#include <limits.h>

#define INDEX(row, col, num_cols) ((row) * (num_cols) + (col))

// array niceties
void arrcpy(int* dest, int* tar, int len) {
	for (int i=0; i<len; i++) {
		dest[i] = tar[i];
	}
}

void printarr_int(int* arr, int len) {
	for (int i=0; i<len; i++) {
		printf("%d ", arr[i]);
	}
	printf("\n");
}

void printarr_double(double* arr, int len) {
	for (int i=0; i<len; i++) {
		printf("%3.2f ", arr[i]);
	}
	printf("\n");
}

int arrdiff(int* arr1, int* arr2, int size) {
	int total = 0;
	for (int i = 0; i < size; i++) {
		total += arr1[i] == arr2[i] ? 0 : 1;
	}
	return total;
}

// convolutional stuff
// Using a 2, 1, 2 decoder from lecture slides

typedef struct {
	int bit;
	int num_errors;
	int last_state;
} node;

int state_outputs[8][2] = {
	{0, 0}, // s0
	{1, 1},
	{1, 1}, // s1
	{0, 0},
	{1, 0}, // s2
	{0, 1},
	{0, 1}, // s3
	{1, 0},
};

int last_states[4][2] = {
	{0, 1},
	{2, 3},
	{0, 1},
	{2, 3},
};

void print_node(node this_node) {
	if (this_node.num_errors > INT_MAX / 2 - 100) {
		printf("INF|%d\t", this_node.bit);
	} else if (this_node.num_errors < -1 * INT_MAX / 2 + 100) {
		printf("-INF|%d\t", this_node.bit);
	} else {
		printf("%d|%d\t", this_node.num_errors, this_node.bit);
	}
}

void print_trellis(node *trellis, int size) {
	for (int state=0; state<4; state++) {
		for (int i=0; i<size; i++) {
			node this_node = trellis[INDEX(i, state, 4)];
			print_node(this_node);
			// printf("%d.%d\t", trellis[INDEX(i, state, 4)].num_errors, trellis[INDEX(i, state, 4)].bit);
		}
		printf("\n");
	}
}

int convolution_errors(long long ndata, int ngroup, double n_0, int seed_offset) {
	// steps:
	// Encoding:
	//   Generate `ngroup` random points
	//   zero-pad and initialize shift register (probably an int)
	//   generate output 2d-array of -1 or +1 doubles
	//
	// Decoding:
	//   figure it out later
	//

	unsigned int seed;

	if (-1 == (seed = (unsigned int) time((time_t *) NULL) + seed_offset)) {
		fprintf(stderr, "time() failed to set seed");
		exit(1);
	}
	seed_xoro(seed);

	long long nremaining = ndata;
	int errors = 0;

	int total_length = ngroup + 2;
	int *bits = malloc((total_length) * sizeof(int));
	int *received = malloc(total_length * 2 * sizeof(int));
	node *trellis = malloc((total_length) * 4 * sizeof(node));
	int *decoded = malloc(total_length * sizeof(int));
	while (nremaining) {
		ngroup = ngroup < nremaining ? ngroup : nremaining;
		nremaining -= ngroup;
		int memory[2] = {0, 0};

		for (int i = 0; i < total_length; i++) {
			// generate random bits and then zero-pad
			if (i < ngroup) {
				if (ngroup == 5) {
					// debug message stuff
					int message[5] = {0, 1, 1, 0, 1};
					bits[i] = message[i]; // rand() & 1;
				} else {
					bits[i] = rand() & 1;
				}
			} else {
				bits[i] = 0;
			}
			// add the noise
			double coded_1 = (double)((bits[i] ^ memory[0] ^ memory[1]) * 2 - 1);
			double coded_2 = (double)((bits[i] ^ memory[1]) * 2 - 1);
			coded_1 += gaussian_noise() * sqrt(n_0/2);
			coded_2 += gaussian_noise() * sqrt(n_0/2);

			int idx1 = INDEX(i, 0, 2);
			int idx2 = idx1 + 1;
			received[idx1] = coded_1 > 0 ? 1 : 0;
			received[idx2] = coded_2 > 0 ? 1 : 0;

			// shift the stuff
			memory[1] = memory[0];
			memory[0] = bits[i];
		}
		// message is generated!
		// printarr_int(bits, total_length);
		for (int i = 0; i < total_length; i++) {
			// printarr_int(&received[INDEX(i, 0, 2)], 2);
		}

		trellis[INDEX(0, 0, 4)].num_errors = 0;
		for (int i=1; i<total_length * 4; i++) {
			trellis[i].num_errors = INT_MAX / 2; // divide by 2 to avoid int wrap errors
		}
		int decode_bits[2] = {0};
		// do first two manually because it is a special case
		arrcpy(decode_bits, &received[INDEX(0, 0, 2)], 2);
		
		trellis[INDEX(0, 0, 4)].num_errors = arrdiff(decode_bits, state_outputs[0], 2);
		trellis[INDEX(0, 0, 4)].bit = 0;
		trellis[INDEX(0, 2, 4)].num_errors = arrdiff(decode_bits, state_outputs[1], 2);
		trellis[INDEX(0, 2, 4)].bit = 1;


		// loop for the rest of the trellis
		for (int i=1; i<total_length; i++) {
			arrcpy(decode_bits, &received[INDEX(i, 0, 2)], 2);
			for (int state=0; state<4; state++) {
				trellis[INDEX(i, state, 4)].num_errors = INT_MAX;
				for (int k=0; k<2; k++) {
					int last_state = last_states[state][k];
					int diff_errors = arrdiff(decode_bits, state_outputs[INDEX(last_state, state / 2, 2)], 2);
					int last_errors = trellis[INDEX(i-1, last_state, 4)].num_errors;
					int test_errors = diff_errors + last_errors;
					// printf("%d %d %d - %d - %d + %d = %d\n", i, state, k, last_state, diff_errors, last_errors, test_errors);
					if (test_errors < trellis[INDEX(i, state, 4)].num_errors) {
						trellis[INDEX(i, state, 4)].num_errors = test_errors;
						trellis[INDEX(i, state, 4)].bit = state / 2; // shorthand for bit is 1 if state >= 2
						trellis[INDEX(i, state, 4)].last_state = last_state;
					}
					// TODO: if test_errors == this_errors then consider last errors for picking which line to create
				}
			}
		}
		// print_trellis(trellis, total_length);
		// trellis is filled out, time to reconstruct the message
		int this_state;
		int fewest_errors = INT_MAX;
		// find the best finishing state
		for (int state=0; state<4; state++) {
			int num_errors = trellis[INDEX(total_length-1, state, 4)].num_errors;
			if (num_errors < fewest_errors) {
				fewest_errors = num_errors;
				this_state = state;
			}
		}
		// printf("RESULT:\n");
		for (int i=total_length-1; i>=0; i--) {
			// print_node(trellis[INDEX(i, this_state, 4)]);
			// printf("\n");
			decoded[i] = trellis[INDEX(i, this_state, 4)].bit;
			// walk back through the states
			this_state = trellis[INDEX(i, this_state, 4)].last_state;
		}
		// printf("DECODED: ");
		// printarr_int(decoded, total_length);
		errors += arrdiff(decoded, bits, ngroup);
	}

	free(decoded);
	free(bits);
	free(received);
	free(trellis);
	return errors;
}






// hamming stuff

int hash(int* arr, int arr_length) {
	int total = 0;
	for (int i = 0; i< arr_length; i++) {
		total += arr[i] << (arr_length - i - 1);
	}
	return total;
}

int codewords[16][7] = {
	{0, 0, 0, 0, 0, 0, 0}, // 0
	{0, 0, 0, 1, 1, 1, 1}, // 1
	{0, 0, 1, 0, 1, 1, 0}, // 2
	{0, 0, 1, 1, 0, 0, 1}, // 3
	{0, 1, 0, 0, 1, 0, 1}, // 4
	{0, 1, 0, 1, 0, 1, 0}, // 5
	{0, 1, 1, 0, 0, 1, 1}, // 6
	{0, 1, 1, 1, 1, 0, 0}, // 7
	{1, 0, 0, 0, 0, 1, 1}, // 8
	{1, 0, 0, 1, 1, 0, 0}, // 9
	{1, 0, 1, 0, 1, 0, 1}, // 10
	{1, 0, 1, 1, 0, 1, 0}, // 11
	{1, 1, 0, 0, 1, 1, 0}, // 12
	{1, 1, 0, 1, 0, 0, 1}, // 13
	{1, 1, 1, 0, 0, 0, 0}, // 14
	{1, 1, 1, 1, 1, 1, 1}, // 15
};

int hash_to_bucket[257] = {-1};

void init_hash() {
	for (int i=0; i<16; i++) {
		hash_to_bucket[hash(codewords[i], 7)] = i;
	}
}

int h_t[8][3] = {
	{0, 1, 1}, // 0
	{1, 0, 1}, // 1
	{1, 1, 0}, // 2
	{1, 1, 1}, // 3
	{1, 0, 0}, // 4
	{0, 1, 0}, // 5
	{0, 0, 1}, // 6
};


int error_idx(int* arr) {
	// arr is received, need to find errors
	int s[3] = {0};
	for (int i = 0; i<3; i++) {
		for (int j = 0; j<7; j++) {
			s[i] += arr[j] * h_t[j][i];
		}
		s[i] = s[i] % 2;
	}

	switch (hash(s, 3)) {
		case 1:
			return 6;
		case 2:
			return 5;
		case 4:
			return 4;
		case 7:
			return 3;
		case 6:
			return 2;
		case 5:
			return 1;
		case 3:
			return 0;
		default:
			return -1;
	}
}

void data_to_bits(int* dest, int n, int size) {
	for (int i = size-1; i >= 0; i--) {
		dest[i] = n % 2;
		n /= 2;
	}
}

int uncoded_errors(long long ndata, double n_0, int seed_offset) {
	int i;
	unsigned int seed;

	if (-1 == (seed = (unsigned int) time((time_t *) NULL) + seed_offset)) {
		fprintf(stderr, "time() failed to set seed");
		exit(1);
	}
	seed_xoro(seed);

	int errors = 0;
	int bit;
	double received;

	for (i = 0; i < ndata; i++) {
		bit = rand() & 1;
		received = (double)bit * 2 - 1; // convert to -1 or 1
		received += gaussian_noise() * sqrt(n_0/2);
		errors += (received > 0 && bit) || (received <= 0 && !bit) ? 0 : 1;
	}
	return errors;
}

int hamming_errors(long long ndata, double n_0, int seed_offset) {
	int i;
	int j;
	unsigned int seed;

	// if (ndata % 4 != 0) {
	// 	fprintf(stderr, "ndata must be divisible by 4");
	// 	exit(1);
	// }

	if (-1 == (seed = (unsigned int) time((time_t *) NULL) + seed_offset)) {
		fprintf(stderr, "time() failed to set seed");
		exit(1);
	}
	srand(seed);

	int errors = 0;
	int error;
	int data = 0;
	int bits[4] = {0};
	int codeword[7] = {0};

	int data_received = 0;
	int bits_received[4] = {0};
	double received[7] = {0};
	int codeword_received[7] = {0};
	for (i = 0; i < ndata/4; i++) {
		// generate the random bits
		data = rand() % 16;
		data_to_bits(bits, data, 4);
		
		arrcpy(codeword, codewords[hash(bits, 4)], 7);
		// printarr_int(codeword, 7);
		// map to -1 and 1 and store as doubles
		for (j=0; j<7; j++) {
			received[j] = (double)(codeword[j] * 2 - 1);
		}
		for (j=0; j<7; j++) {
			received[j] += gaussian_noise() * sqrt(n_0/2);
		}
		// printarr_double(received, 7);

		for (j=0; j<7; j++) {
			codeword_received[j] = received[j] > 0 ? 1 : 0;
		}
		// printarr_int(codeword_received, 7);
		error = error_idx(codeword_received);
		// printf("ERROR AT: %d\n", error);
		if (error >= 0) {
			codeword_received[error] += 1;
			codeword_received[error] %= 2;
		}
		// printarr_int(codeword_received, 7);
		data_received = hash_to_bucket[hash(codeword_received, 7)];

		// printf("%d\n", data_received);
		data_to_bits(bits_received, data_received, 4);
		// printarr_int(bits_received, 4);

		errors += arrdiff(bits, bits_received, 4);
		
		// printf("\n\n");
	}

	return errors;
}
