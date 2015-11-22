#include <stdio.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "../lib/helper_cuda.h"

#define ROWS 10
#define COLS 10

int main(){
	
	//initilaize matrix
    float *h_A = (float *)malloc(ROWS * COLS * sizeof(float));
    int i,j;
	
	printf("input matrix\n");
	for(i = 0; i < ROWS; i++){
        for(j = 0; j < COLS; j++){
            h_A[i + j*ROWS] = i+j;
			printf("%f ",h_A[i + j*ROWS]);
		}
		printf("\n");
	}
	printf("\n\n");

	//allocate and copy to GPU
    float *d_A;           
	checkCudaErrors(cudaMalloc(&d_A, ROWS * COLS * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_A, h_A, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice));
	
	// host side SVD results space
    float *h_U = (float *)malloc(ROWS * ROWS * sizeof(float));
    float *h_V = (float *)malloc(COLS * COLS * sizeof(float));
    float *h_S = (float *)malloc(min(ROWS, COLS) * sizeof(float));
	
    // --- device side SVD workspace and matrices
    float *d_U,*d_V,*d_S;            
	checkCudaErrors(cudaMalloc(&d_U,  ROWS * ROWS     * sizeof(float)));       
	checkCudaErrors(cudaMalloc(&d_V,  COLS * COLS     * sizeof(float)));           
	checkCudaErrors(cudaMalloc(&d_S,  min(ROWS, COLS) * sizeof(float)));

	//cuSolver initialization
	cusolverDnHandle_t handle = NULL;
	checkCudaErrors(cusolverDnCreate(&handle));
	int worksize;
	cusolverDnSgesvd_bufferSize(handle, ROWS, COLS, &worksize ); 
	float *work;   
	checkCudaErrors(cudaMalloc(&work, worksize * sizeof(float)));

	// SVD execution
	int *devInfo;           
	checkCudaErrors(cudaMalloc(&devInfo, sizeof(int)));
	checkCudaErrors(cusolverDnSgesvd (handle, 'A', 'A', ROWS, COLS, d_A, ROWS, d_S, d_U, ROWS, d_V, COLS, work, worksize, NULL, devInfo));
    int devInfo_h = 0;  
	checkCudaErrors(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo_h != 0) {
		fprintf(stderr,"Error in svd execution\n");
	}

	// --- Moving the results from device to host
    checkCudaErrors(cudaMemcpy(h_S, d_S, min(ROWS, COLS) * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_U, d_U, ROWS * ROWS     * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_V, d_V, COLS * COLS     * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Singular values\n");
    for(i = 0; i < min(ROWS,COLS); i++) 
		printf("%f ",h_S[i]);
	printf("\n\n");	
	
    printf("\nLeft singular vectors - For y = A * x, the columns of U span the space of y\n");
    for(i= 0; i < ROWS; i++) {
        printf("\n");
        for(j = 0; j < ROWS; j++)
            printf("%f ",h_U[j*ROWS + i]);
    }
	printf("\n\n");	
	
    printf("\nRight singular vectors - For y = A * x, the columns of V span the space of x\n");
    for(i= 0; i < COLS; i++) {
        printf("\n");
        for(j = 0; j < COLS; j++)
            printf("%f ",h_V[j*ROWS + i]);
    }
    
	
	
	
	cusolverDnDestroy(handle);
	
	
	return 0;
}
