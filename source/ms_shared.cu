/**
 * File: ms_shared.cu
 * Description:
 * 	A cuda implemented mean shift algorithm. Takes advantage of the gpu's
 *	shared memory.
 * Authors:
 * 	Georgios Kiloglou,	AEM: 8748,	georkilo@auth.gr
 *	Nikolaos Katomeris,	AEM: 8551,	ngkatomer@auth.gr
 *
 * Date:
 *	January, 2018
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

/**
 * Constant Value definitions for the cuda function mean_shift():
 * -N: is the number of points/elements
 * -d: is the number of dimensions/attributes
 * -H: is the radius of the circle, the points in which will
 *		"shift" the mean value from the center of the circle
 *		to the mean of all these points.
 * -E: if the mean is "shifted" less than E the algorithm
 *		stops as the mean has converged to the margin we need.
 */
#ifdef SEED_DATASET
	#define N 210
	#define d 7
	#define H 3
	#define E 1e-9

	const char* input_data_filename = "../data/seeds_x.bin";
	const char* ver_data_filename = "../data/seeds_y_H3.00.bin";
#else
	#define N 600
	#define d 2
	#define H 1 // H=sigma^2
	#define E 1e-9

	const char* input_data_filename = "../data/r15_x.bin";
	const char* ver_data_filename = "../data/r15_y_H1.bin";
#endif

// The max acceptable error between the results and the matlab's algorithm results
#define MAX_VER_ERROR 1e-4

/**
 * Allocates continuous memory for a 2D double array.
 */
double** allocate2DDArray(int rows, int cols);
/**
 * Prints a 2D double array.
 */
void print(double** array, int rows, int cols);
/**
 * Cuda function:
 * Calculates the shifted mean <y> of each point of <x>
 * using the N, d, H and E defined values.
 */
 __global__ void mean_shift(double* x, double* y);

int main(int argc, char** argv)
{
	double **x, **y;
	x = allocate2DDArray(N, d);
	y = allocate2DDArray(N, d);
	struct timespec start, finish;
	
	//---Array initialization in ram--
	if (x == NULL || y == NULL){
		printf("Error at allocating memory for table x or y\n");
		exit(1);
	}
	
	FILE* fp = fopen(input_data_filename, "rb");
	if (fp == NULL){
		printf("Couldn't open input file\n");
		exit(2);
	}
	
	if(fread(x[0], sizeof(double), N*d, fp) != N*d){
		printf("Error at reading the input_file's data");
		exit(3);	
	}
	fclose(fp);
	// Copy the x data to y (means are initialized to points)
	memcpy(y[0], x[0], N*d*sizeof(double));
	
	//---GPU memory init-------
	double *dev_x, *dev_y;
	cudaMalloc(&dev_x, N*d*sizeof(double));
	cudaMalloc(&dev_y, N*d*sizeof(double));
	cudaMemcpy(dev_x, x[0], N*d*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, y[0], N*d*sizeof(double), cudaMemcpyHostToDevice);
	//-------------------------

	clock_gettime(CLOCK_REALTIME, &start);
	
	mean_shift<<<N, N>>>(dev_x, dev_y);
	
	cudaDeviceSynchronize();
	
	//---Get time--------------
	clock_gettime(CLOCK_REALTIME, &finish);
        long seconds = finish.tv_sec - start.tv_sec; 
        long nano_seconds = finish.tv_nsec - start.tv_nsec; 
         
        if (start.tv_nsec > finish.tv_nsec) {
        seconds--; 
        nano_seconds += 1e9; 
        } 
        printf("_time: %lf\n", seconds + (double)nano_seconds/1e9);
	//-------------------------
	
	// Get the calculated values and free the gpu memory
	cudaMemcpy(y[0], dev_y, N*d*sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(dev_x);
	cudaFree(dev_y);
	
	
	// Verify them
	int verify = 1;
	double abs_diff = 0;
	fp = fopen(ver_data_filename, "rb");
	if (fp == NULL){
		printf("Couldn't open verification file\n");
		verify = 0;	
	}
	if(fread(x[0], sizeof(double), N*d, fp) != N*d){
		printf("Error at reading the ver_file's data");
		verify = 0;
	}
	fclose(fp);
	if (verify){
		int i, j;
		double dis;
		for (i = 0; i < N; i++){
			for (j = 0; j < d; j++){
				dis = fabs(y[i][j] - x[i][j]);
				if(dis > MAX_VER_ERROR){
					printf("Error %lf at y[%d][%d]\n", dis, i, j);
				}
				abs_diff += dis;
			}
		}
		printf("Verification sum = %lf\n", abs_diff);
	}
	//DEBUG: Print them
	//print(x, N, d);
	
	//---Clean up and return---
	free(x[0]);
	free(y[0]);
	free(x);
	free(y);

	return 0;
}

 __global__ void mean_shift(double* x, double* y){
	int bid = blockIdx.x*d;
	int tid = threadIdx.x*d;
	
	/**
	 * Allocate shared memory for the numerator's and the
	 * denominator's addends.
	 *
	 * An int is also allocated in order for the thread 0 to
	 * inform the other threads of the block that the algorithm
	 * converged so they can break out of the loop.
	 */
	__shared__ double t_den[N]; 
	__shared__ double t_num[N*d];
	__shared__ int converged;
	
	int i,j;
	double multiplier;
	double factor;
	do{
		multiplier = 0;
		//---Each thread calculates its respecting addends---
		for(i = 0; i < d; i++){
			factor = y[bid+i]-x[tid+i];
			multiplier += factor*factor;
		}
		// shift_distancetance = sqrt(multiplier) and 
		// if (shift_distancetance > H) we zero the multiplier of this point
		// as it shouldn't affect the new mean.
		// shift_distancetance > H ==> shift_distancetance^2 > H^2, to not use sqrt()
		// Because most points have shift_distancetance > H*H we avoid 
		// 	accessing the global memory for these points by zeroing
		// 	the t_den and t_num values immediately in that case.
		if (multiplier > H*H){
		  t_den[threadIdx.x] = 0;
		  for (i = 0; i < d; i++)
		    t_num[tid+i] = 0;
		}
		else{
			multiplier = exp(-multiplier/(2*H));
			t_den[threadIdx.x] = multiplier;
			
			for(i = 0; i < d; i++){
				t_num[tid+i] = multiplier*x[tid+i];
			}
		}
		//---------------------------------------------------
		// Wait untill all the addends are calculated
		__syncthreads();

		//Now the sum can be performed.
		//The zero id thread performs the sum---
		if (threadIdx.x == 0){
			double y_next[d] = {0};
			double den = 0;
			converged = 0;
			// Sum the denominator's addends
			for (i = 0; i < N; i++){
				den += t_den[i];
			}
			// Should not divide with 0
			// If that's the case the point won't have any other points
			// in range (h) so it has already converged
			if (den == 0){
				break;
			}
			// Sum the numerator's addends
			for(i = 0; i < d*N; i+=d){
				for (j = 0; j < d; j++){
					y_next[j]+=t_num[i+j];
				}
			}
			// Divide, calculate the "shift" and update the y (mean value)
			double shift_distance = 0;
			for(i = 0; i < d; i++){
				y_next[i] = y_next[i]/den;
				shift_distance += (y_next[i]-y[bid+i])*(y_next[i]-y[bid+i]);
				y[bid+i] = y_next[i];
			}
			// If the distance is less than E, the algorithm will stop.
			converged = (shift_distance < E) ? 1 : 0;
		}
		// Make sure the threads wait for the zero thread to finish, before
		// continuing (or returning).
		__syncthreads();
	}while(!converged);
}

void print(double** array, int rows, int cols)
{
	int i, j;
	for (i = 0; i < rows; i++){
		for (j = 0; j < cols; j++){
			printf("%.12lf ", array[i][j]);
		}
		printf("\n");
	}
}

double** allocate2DDArray(int rows, int cols)
{
	double* data = (double*) malloc(rows*cols*sizeof(double));
	if (data == NULL) return NULL;
	double **table = (double **)malloc(rows*sizeof(double));
	if (table == NULL) return NULL;
	int i;
	for (i = 0; i < rows; i++)
	{
		table[i] = data+cols*i;
	}
	return table;
}

