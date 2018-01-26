/**
 * File: ms_global.cu
 * Description:
 * 	A cuda implemented mean shift algorithm.
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

double** allocate2DDArray(int rows, int cols);
void print(double** array, int rows, int cols);

/**
 * The mean_shift function (exact same algorithm with) the one described
 * in the ms_shared.cu with the difference that the numerator's and
 * denominator's addends reside in the global memory of the gpu.
 */
__global__ void mean_shift(double* x, double* y, double* t_den, double* t_num){
	int bid = blockIdx.x*d;
	int tid = threadIdx.x*d;
	int gid = bid*N+tid;
	
	__shared__ int converged;
	
	int i; 
	double multiplier;
	double factor;
	do{
		//---Each thread calculates its respecting addends---
		multiplier = 0.0;
		for(i = 0; i < d; i++){
			factor = y[bid+i]-x[tid+i];
			multiplier += factor*factor;
		}
		
		if (multiplier > H*H){
		  t_den[blockIdx.x*N+threadIdx.x] = 0;
		  for (i = 0; i < d; i++)
		    t_num[gid+i] = 0;
		}
		else{
			multiplier = exp(-multiplier/(2*H));
			t_den[blockIdx.x*N+threadIdx.x] = multiplier;
			for(i = 0; i < d; i++){
				t_num[gid+i] = multiplier*x[tid+i];
			}
		}
		//---------------------------------------------------
		__syncthreads();

		//---The zero id threads perform the sum---
		
		if (threadIdx.x == 0){
			double y_next[d] = {0};
			double den = 0;
			converged = 0;
			
			for (i = 0; i < N; i++){
				den += t_den[blockIdx.x*N+i];
			}
			if (den == 0){
				//Surely converged
				break;
			}
			int j;
			for(i = 0; i < d*N; i+=d){
				for (j = 0; j < d; j++){
					y_next[j]+=t_num[gid+i+j];
				}
			}
			double dis = 0;
			for(i = 0; i < d; i++){
				y_next[i] = y_next[i]/den;
				dis += (y_next[i]-y[bid+i])*(y_next[i]-y[bid+i]);
				y[bid+i] = y_next[i];
			}

			converged = (dis < E) ? 1 : 0;
		}
		__syncthreads();
	}while(!converged);
}

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
		printf("Couldn't open file\n");
		exit(2);
	}
	
	if(fread(x[0], sizeof(double), N*d, fp) != N*d){
		printf("Error at reading file data");
		exit(3);	
	}
	fclose(fp);
	memcpy(y[0], x[0], N*d*sizeof(double));
	
	//---GPU memory init-------
	double *dev_x, *dev_y, *den, *num;
	cudaMalloc((void**)&dev_x, N*d*sizeof(double));
	cudaMalloc((void**)&dev_y, N*d*sizeof(double));
	cudaMalloc((void**)&den, N*N*sizeof(double));
	cudaMalloc((void**)&num, N*N*d*sizeof(double));
	cudaMemcpy(dev_x, x[0], N*d*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, y[0], N*d*sizeof(double), cudaMemcpyHostToDevice);
	//-------------------------
	
	clock_gettime(CLOCK_REALTIME, &start);

	mean_shift<<<N, N>>>(dev_x, dev_y, den, num);
	
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
	cudaFree(num);
	cudaFree(den);
	
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
	
	//---Clean up---
	free(x[0]);
	free(y[0]);
	free(x);
	free(y);
	return 0;
}

void print(double** array, int rows, int cols)
{
	int i, j;
	for (i = 0; i < rows; i++){
		for (j = 0; j < cols; j++){
			printf("%lf ", array[i][j]);
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
