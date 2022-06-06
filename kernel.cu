#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "timer.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// Algorithm settings
#define MAX_ITER 1000
#define TOLERANCE 0.0001
#define NN 8192
#define NM 8192
#define gridSize 128
#define blockSize 512

// CUDA kernel
__global__ void jacob(float* A,float* A_new, int nx, int ny,float *max)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; // global thread id
    int stride = blockDim.x*gridDim.x;
    int row = index / ny;
    int diff = index - (row * ny);
    int k = (row + 1) * (ny + 2) + diff + 1;
    float tmax = 0;
    while (index < nx * ny) {
        A_new[k] = 0.25 * (A[k - 1] + A[k + 1] +A[k - (ny + 2)] + A[k + (ny + 2)]);
	float update = 0.0;
	if(A_new[k]-A[k]>0){
		update = A_new[k]-A[k];
	}else{
		update = A[k]-A_new[k];
	}
	if(update>tmax)tmax = update;
	row = index / ny;
	index += stride;
    	diff = index - (row * ny);
    	k = (row + 1) * (ny + 2) + diff + 1;
    }
    int thIdx = threadIdx.x;
    __shared__ float shArr[blockSize];
    shArr[thIdx] =tmax;
    __syncthreads();
    for(int size = blockSize/2;size>0;size/=2){
    	if(thIdx<size)
		shArr[thIdx]= fmax(shArr[thIdx+size],shArr[thIdx]);
	__syncthreads();
    }
    max[blockIdx.x] = shArr[0];

}
__global__ void gmax(float* A,float *max){
    int thIdx = threadIdx.x;
    float tmax = A[thIdx];
    __shared__ float shArr[blockSize];
    shArr[thIdx] =tmax;
    __syncthreads();
    for(int size = blockSize/2;size>0;size/=2){
        if(thIdx<size)
                shArr[thIdx]= fmax(shArr[thIdx+size],shArr[thIdx]);
        __syncthreads();
    }
    max[blockIdx.x] = shArr[0];
}

int main(int argc, char* argv[]) {
    int n = NN;
    int m = NM;
    int size=n*m;
    //allocate memory
    float* host_A = (float*)malloc(sizeof(float) * size);
    float* host_Anew = (float*)malloc(sizeof(float) * size);
    //initi
    
    for (int i = 0; i < size; i++) {
	if(i%m==0){
		host_A[i] = 1.0;
        	host_Anew[i] = 1.0;
	}else{
		host_A[i] = 0;
        	host_Anew[i] = 0;
	}
    }

    //  CUDA
    float* device_A, * device_Anew;
    cudaMalloc(&device_A, size * sizeof(float));
    cudaMalloc(&device_Anew, size * sizeof(float));

    //set parameter
    float *terr;   
    cudaMalloc(&terr,gridSize*sizeof(float));
    //start timer
    StartTimer();
    int iter = 0;
    cudaMemcpy(device_A, host_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_Anew, host_A, size * sizeof(float), cudaMemcpyHostToDevice);
    float error = 0;
    for (iter = 0; iter < MAX_ITER; iter++) {
        error = 0;
	//cuda calculation
        //setup d_ value
	//cudaMemcpy(device_Anew, host_A, size * sizeof(float), cudaMemcpyHostToDevice);
        float*tmp = device_A;
	device_A = device_Anew;
        device_Anew = tmp;
	cudaMemset(&terr,0.0,gridSize*sizeof(float));
	//run
        jacob << <gridSize, blockSize >> > (device_A, device_Anew, n - 2, m - 2,terr);
        gmax <<<1,blockSize>>> (terr,terr);
	//output value to find error
 //       cudaMemcpy(host_Anew, device_Anew, size * sizeof(float), cudaMemcpyDeviceToHost);
 	cudaMemcpy(&error, terr, sizeof(float), cudaMemcpyDeviceToHost);
        //find error
        if (iter%100==0) {
            printf("%5d, %f\n", iter, error);
        }
        if(error<TOLERANCE)break;
    }
    double runtime = GetTimer();
    printf(" total: %f s\n", runtime / 1000);
    //free memory
    free(host_A);
    free(host_Anew);
    cudaFree(device_A);
    cudaFree(device_Anew);
    return 0;
}
