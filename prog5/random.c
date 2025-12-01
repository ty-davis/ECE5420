#include <stdio.h>
#include <stdlib.h>
#include "frannor.h"
#include <time.h>

int main(int argc, char **argv)
{
	double *noise=NULL;
	int i;
	int ndata;
	unsigned int seed;

	if(argc!=2)
	{
		fprintf(stderr, "usage : random 5");
		exit(0);
	}

	ndata = atoi(argv[1]);

	if (-1 == (seed = (unsigned int) time((time_t *) NULL)))
	{
		fprintf(stderr, "time() failed to set seed");
		exit(0);
	}

	srannor(seed);  //seed random number generator
	noise = (double*) calloc (ndata, sizeof(double));
	for(i=0; i<ndata; i++)  //create random data
	{
		/* Compute noise vector elements in [-1, 1] */
		/* GAUSS METHOD. frannor gives elements in N(0,1)--ie. pos & negs */
		noise[i] = (double) frannor();
	}

	printf("[");
	for(i=0; i<ndata; i++)
		printf("%3.2f, ", noise[i]);
	printf("]\n");

	free(noise);  //free allocated memory
	return(1);
}
