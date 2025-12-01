#include <stdio.h>
#include <stdlib.h>
#include "frannor.h"
#include <time.h>
#include <math.h>

/*
* Steps:
* Assert ndata divisible by 4.
* Iterate over ndata/4 elements.
*    Generate random sequence of 4 bits
*    Get codeword from bits
*    Convert to BPSK
*    Add noise
*    Decode
*    Find syndrome
*    Fix error
*    Get bits from codeword
*    Compare and add to total errors
*/

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

int arrdiff(int* arr1, int* arr2, int size) {
	int total = 0;
	for (int i = 0; i < size; i++) {
		total += arr1[i] == arr2[i] ? 0 : 1;
	}
	return total;
}


int main(int argc, char *argv[]) {
	int i;
	int j;
	int k;
	unsigned int seed;

	if (argc != 3) {
		fprintf(stderr, "usage: channel_coding <int:ndata> <double:N_0>");
		exit(1);
	}

	int ndata = atoi(argv[1]);
	double n_0 = atof(argv[2]);

	init_hash();

	if (ndata % 4 != 0) {
		fprintf(stderr, "ndata must be divisible by 4");
		exit(1);
	}

	if (-1 == (seed = (unsigned int) time((time_t *) NULL))) {
		fprintf(stderr, "time() failed to set seed");
		exit(1);
	}
	srannor(seed);
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
	for (i=0; i<ndata/4; i++) {
		// generate the random bits
		data = rand() % 16;
		data_to_bits(bits, data, 4);
		// for (j=0; j<4; j++) {
		// 	bits[j] = rand() & 1;
		// }
		
		arrcpy(codeword, codewords[hash(bits, 4)], 7);
		// printarr_int(codeword, 7);
		// map to -1 and 1 and store as doubles
		for (j=0; j<7; j++) {
			received[j] = (double)(codeword[j] * 2 - 1);
		}
		for (j=0; j<7; j++) {
			received[j] += (double)frannor() * sqrt(n_0/2);
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
	printf("%d errors", errors);

	return 0;
}
