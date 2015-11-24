// 
 
#include <stdio.h>
#include <cuda.h>

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>


#include <fcntl.h>    /* For O_RDWR */
#include <unistd.h>   /* For open(), creat() */
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <fstream>

#include <sys/time.h>
#include <time.h>

#include <string.h>
#include <assert.h>

#include <cula_lapack.h>

#include <cula_lapack_device.h>
#include <cublas.h>
#include "lib/helpers.cuh"

#define imin(X, Y)  ((X) < (Y) ? (X) : (Y))
//void checkStatus(culaStatus status);

using namespace Eigen;
using namespace std;
using Eigen::ArrayXXd ;
using Eigen::MatrixXd ;

#include "common.h"
#include <cublas_v2.h>
#include "lib/myTimer.h"


#include <cuda_runtime.h>
#include <cusolverDn.h>

cublasHandle_t handle;

__global__ 
void applyCube(float *original,float * computed,float * cubeDerivation, int n,int p) 
{
	
	//converted to 2d
	
	int y = blockIdx.y*blockDim.y+threadIdx.y; //n
	int x = blockIdx.x*blockDim.x+threadIdx.x; //p
	
	if(x<p && y<n){
		float originalFloat = original[y*p+x];
		float tmp = originalFloat*originalFloat;
		cubeDerivation[y*p+x]=3*tmp;
		computed[y*p+x] =tmp*originalFloat;
	}
	
}



__global__ 
void findMean(float * computed,float * alternative,int n,int p) 
{
	
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	
	computed[x]=alternative[x]/p;
	
}


__global__ 
void multiplyColumnVise(float *original,float * calculated,float * factor,int n,int p) 
{
	
	//int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y; //n
	int x = blockIdx.x*blockDim.x+threadIdx.x; //p
	//int i;
	
	if(y<n && x<p){
		float mulFactor;
		int index;
		
		mulFactor = factor[y];
		index = x*n+y;
		calculated[index] = original[index]*mulFactor;

	}	
}


__global__ 
void subtractMatrices(float * target,float *A,float * B,int n) 
{
	
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	if(x<n*n){
		target[x] = A[x]-B[x];
	}
		
}



/*
BLAS functions
*/


void create_blas_handler(){
	
	// Create a handle for CUBLAS
	cublasCreate(&handle);
	
}

void destroy_blas_handler(){
	
	 // Destroy the handle
	cublasDestroy(handle);
	
}


// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(float *A, float *B, float *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;
	
	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	cudaDeviceSynchronize();
	
}


// Improved version for transpose multiplication
void gpu_blas_mmulImprove(float *A, float *B, float *C, const int m, const int k, const int n,const float p_) {
	int lda=m,ldb=k,ldc=m;
	const float alf = 1/p_;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;
	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	cudaDeviceSynchronize();
}

__global__
void memSetInCuda(float *d_singleArray,float num,int sizeofSingleArray){
	
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	if(x<sizeofSingleArray){
		
		d_singleArray[x] = num;
	}
	
}

int runSVDonCUDA(MatrixXd& input,VectorXd& singularValue,MatrixXd& singularVectors,int ROWS,int COLS){
	//initilaize matrix
	//int ROWS =1000;
	//int COLS =3;
	MatrixXf DataMatrix = input.cast<float>();
	
    float* h_A = DataMatrix.data();
	
	
	int M = ROWS;
    int N = COLS;
	
	culaStatus status;
	
	/* Setup SVD Parameters */
    int LDA = M;
    int LDU = M;
    int LDVT = N;
   
    float* A = NULL;
    float* S = NULL;
    float* U = NULL;
    float* VT = NULL;
	
	time_t begin_time;
    time_t end_time;
    int cula_time;

    char jobu = 'N';
    char jobvt = 'A';
	
			//cout<<"checking h_A 1 "<<h_A[0]<<" "<<h_A[1]<<" "<<h_A[1]<<endl;
	
	cudaMalloc((void**)&A, M*N*sizeof(float ));checkCudaError();
			//cout<<"checking h_A 2 "<<h_A[0]<<" "<<h_A[1]<<" "<<h_A[1]<<endl;
	cudaMemcpy( A, h_A,  M*N*sizeof(float ), cudaMemcpyHostToDevice );checkCudaError();
			//cout<<"checking h_A 3 "<<h_A[0]<<" "<<h_A[1]<<" "<<h_A[1]<<endl;
			//float * test = (float *)malloc(M*N*sizeof(float ));
			//cudaMemcpy( test, A,  M*N*sizeof(float ), cudaMemcpyDeviceToHost );checkCudaError();
			//cout<<"checking h_A 4 "<<h_A[0]<<" "<<h_A[1]<<" "<<h_A[1]<<endl;
			
			//checking elements
			//cout<<test[0]<<" "<<test[1]<<" "<<test[2]<<endl;
	//sleep(1);
	//A = (float*)malloc(M*N*sizeof(float));
	cudaMalloc( (void**)&S, imin(M,N)*sizeof(float));checkCudaError();
    //S = (float*)malloc(imin(M,N)*sizeof(float));
    //U = (float*)malloc(N*N*sizeof(float));
	cudaMalloc( (void**)&VT, LDVT*N*sizeof(float));checkCudaError();
    //VT = (float*)malloc(LDVT*N*sizeof(float));

	/*
	if(!A || !S  || !VT) 
    {
        free(A);
        //free(U);
        free(S);
        free(VT);

        return EXIT_FAILURE;
    }
	*/
	/* Initialize CULA */
    status = culaInitialize();
    //checkStatus(status);
	 /* Perform singular value decomposition CULA */
    printf("Performing singular value decomposition using CULA ... \n");

    time(&begin_time);
    status = culaDeviceSgesvd(jobu, jobvt, M, N, A, LDA, S, U, LDU, VT, LDVT);
    //checkStatus(status);
    time(&end_time);

	
    cula_time = (int)difftime( end_time, begin_time);
	
	
	
	culaShutdown();
	//copy back data,maybe should move before culashutdown
	float* S_tmp;
	float * VT_tmp;
	S_tmp = (float*)malloc(imin(M,N)*sizeof(float));
	VT_tmp = (float*)malloc(LDVT*N*sizeof(float));
	cudaMemcpy(S_tmp,S,imin(M,N)*sizeof(float),cudaMemcpyDeviceToHost);checkCudaError();
	cudaMemcpy(VT_tmp,VT,LDVT*N*sizeof(float),cudaMemcpyDeviceToHost);checkCudaError();
	
	
	for (int i = 0; i < singularVectors.size(); i++){
		
		*(singularVectors.data() + i) = (double)VT_tmp[i];
	}
	
	for (int i = 0; i < singularValue.size(); i++){
		
		*(singularValue.data() + i) = (double)S_tmp[i];
	}
	
	cudaDeviceSynchronize();checkCudaError();

    return EXIT_SUCCESS;

}
/*
preprocessVariables initializeCudaForPreprocess(MatrixXd& S,preprocessVariables preprocessData,int n,int p){
	//This is transform so row columns interchanged
	timestamp_t prepr1 = get_timestamp();
	
	MatrixXf DataMatrix = S.cast<float>();
	
    preprocessData.h_A = DataMatrix.data();
	
	cout<<"preprocessData.h_A"<<preprocessData.h_A[0]<<" "<<preprocessData.h_A[1]<<" "<<preprocessData.h_A[1]<<endl;

	timestamp_t prepr2 = get_timestamp();
	cout<<"initializeCudaForPreprocess "<<(prepr2 - prepr1) / 1000000.0L<<endl;
	return preprocessData;
}
*/


cudaVar initializeCuda(MatrixXd& W,MatrixXd& X1,MatrixXd& w_init,cudaVar cudaVariables,int n,int p){
	
	//matrix sizes
	
	MatrixXf f_W = W.cast<float>();
	MatrixXf f_X1 = X1.cast<float>();
	MatrixXf f_w_init = w_init.cast<float>();
	
	MatrixXf f_X1Transpose = f_X1.transpose();
	
	const int matsizeX1 = n*p*sizeof(float);
	const int matsizeX1Transpose = p*n*sizeof(float);
	const int matsizeW = n*n*sizeof(float);
	const int matsizeW1 = n*n*sizeof(float);
	
	const int matsizew_init = n*n*sizeof(float);
	const int matsizeProduct = n*p*sizeof(float);
	const int matsizegwtx = n*p*sizeof(float);
	const int matsizeCubeDerivation = n*p*sizeof(float);
	const int matsizeg_wtx = n*1*sizeof(float);
	const int matsizeGwtxIntoXtranspose = n*n*sizeof(float);
	const int matsizeGwtx_into_W = n*n*sizeof(float);
	const int matsizew_init_w_init_T = n*n*sizeof(float);
	const int matsizeEigenValues = n*1*sizeof(float);
	const int matsizeEigenVectors = n*n*sizeof(float);
	const int matsizeEigenRowWise = n*n*sizeof(float);
	
	const int matsizeW1intoWT = n*n*sizeof(float);
	
	const int matsizebw = n*1*sizeof(float);
	const int matsizezw = n*1*sizeof(float);
	
	const int matsizediagonal = n*1*sizeof(float);
	
	const int matsizeit_num = sizeof(int);
	const int matsizerot_num = sizeof(int);
	const int matsizeAnswer = sizeof(float);
	const int matsize_tmp_w_init = n*n*sizeof(float);

	
	//malloc
	cudaMalloc( (void**)&cudaVariables.X1, matsizeX1 );
	cudaMalloc( (void**)&cudaVariables.X1Transpose, matsizeX1Transpose );
	cudaMalloc( (void**)&cudaVariables.W, matsizeW );
	cudaMalloc( (void**)&cudaVariables.W1, matsizeW1 );
	cudaMalloc( (void**)&cudaVariables.w_init, matsizew_init );
	cudaMalloc( (void**)&cudaVariables.product, matsizeProduct );
	cudaMalloc( (void**)&cudaVariables.gwtx, matsizegwtx );
	cudaMalloc( (void**)&cudaVariables.cubeD, matsizeCubeDerivation );
	cudaMalloc( (void**)&cudaVariables.g_wtx, matsizeg_wtx );
	cudaMalloc( (void**)&cudaVariables.g_wtx_X1_transpose, matsizeGwtxIntoXtranspose );
	cudaMalloc( (void**)&cudaVariables.gwtx_into_W, matsizeGwtx_into_W );
	cudaMalloc( (void**)&cudaVariables.w_init_w_init_T, matsizew_init_w_init_T );
	cudaMalloc( (void**)&cudaVariables.eigenValues, matsizeEigenValues );
	cudaMalloc( (void**)&cudaVariables.eigenVectors, matsizeEigenVectors );
	cudaMalloc( (void**)&cudaVariables.eigenRowWise, matsizeEigenRowWise );
	
	cudaMalloc( (void**)&cudaVariables.W1intoWT, matsizeW1intoWT );
	cudaMalloc( (void**)&cudaVariables.diagonal, matsizediagonal );
	cudaMalloc( (void**)&cudaVariables.answer, matsizeAnswer );
	cudaMalloc( (void**)&cudaVariables.tmp_w_init, matsize_tmp_w_init );
	
	
	//malloc
	cudaMalloc( (void**)&cudaVariables.bw, matsizebw );
	cudaMalloc( (void**)&cudaVariables.zw, matsizezw );
	cudaMalloc( (void**)&cudaVariables.it_num, matsizeit_num );
	cudaMalloc( (void**)&cudaVariables.rot_num, matsizerot_num );
	
	const int sizeofSingleArray = p*1*sizeof(float);
	const int sizeofComputedArray = n*1*sizeof(float);
	cudaMalloc( (void**)&cudaVariables.d_singleArray, sizeofSingleArray );
	cudaMalloc( (void**)&cudaVariables.d_computeArray, sizeofComputedArray );
	
	cudaMallocHost((void **) &cudaVariables.hostpointer,matsizeW);
	
	
	
	int blockSize = 512;
    int gridSize = (int)ceil(((float)(p))/blockSize);
	//MemSet in CUDA
	memSetInCuda<<<gridSize, blockSize>>>(cudaVariables.d_singleArray,1.0,p*1*sizeof(float));
	//cudaMemset(&(cudaVariables.d_singleArray),1.0,p*1*sizeof(float));
	
	//pointers to data
	float *dataFromX1 = f_X1.data();
	float *datafromX1transpose = f_X1Transpose.data();
	float *dataFromW = f_W.data();
	float *dataFromW_init = f_w_init.data();
	
	
	
	//copy data to CUDA
	cudaMemcpy( cudaVariables.X1, dataFromX1, matsizeX1, cudaMemcpyHostToDevice );
	cudaMemcpy( cudaVariables.X1Transpose, datafromX1transpose, matsizeX1Transpose, cudaMemcpyHostToDevice );
	cudaMemcpy( cudaVariables.W, dataFromW, matsizeW, cudaMemcpyHostToDevice );
	cudaMemcpy( cudaVariables.w_init, dataFromW_init, matsizew_init, cudaMemcpyHostToDevice );


	cudaDeviceSynchronize();
	
	return cudaVariables;
	
	
	}


void copyBackW_initfromCUDA(MatrixXd& w_init,MatrixXf& tmp,float * from,float * hostpointer,float * tmp_w_init,int n){

	//copy to tmp_w_init in cuda memory
	cudaMemcpy(tmp_w_init,from,sizeof(float)*n*n,cudaMemcpyDeviceToDevice);
	//copy to host pointer 
	cudaMemcpy(hostpointer,tmp_w_init,n * n * sizeof(float),cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < w_init.size(); i++){
		
		*(w_init.data() + i) = (double)hostpointer[i];
	}

	//cout<<"winit"<<endl;
	//cout<<w_init<<endl;

	}

void saveW1inGPU(MatrixXd& W1,cudaVar cudaVariables,int n){
	
	MatrixXf f_W1 = W1.cast<float>();
	float *dataFromW1 = f_W1.data();
	const int matsizeW1 = n*n*sizeof(float);
	
	cudaMemcpy( cudaVariables.W, dataFromW1, matsizeW1, cudaMemcpyHostToDevice );
	cudaDeviceSynchronize();
	
}




void cubeOnGPU(cudaVar cudaVariables,int n,int p){
	float * original=cudaVariables.product;
	float * computed=cudaVariables.gwtx;
	float * cubeDerivation=cudaVariables.cubeD;
	float * g_wtx = cudaVariables.g_wtx;
	
	
	
    dim3 blockSize(16,16);
	dim3 gridSize((int)ceil(((float)p)/blockSize.x),(int)ceil(((float)n)/blockSize.y));
	
    applyCube<<<gridSize, blockSize>>>(original,computed,cubeDerivation,n,p);
	cudaDeviceSynchronize();

	float * d_singleArray = cudaVariables.d_singleArray;
	float * d_computeArray = cudaVariables.d_computeArray;

	gpu_blas_mmul(cubeDerivation, d_singleArray, d_computeArray,n,p,1);
	
	blockSize = n;
    gridSize = 1;
	//original mean function
    
    findMean<<<gridSize, blockSize>>>(g_wtx,d_computeArray,n,p);
    cudaDeviceSynchronize();
		
	}

	
	
void multiplyColumnViseOnGPU(cudaVar cudaVariables,int n,int p){
	float * original=cudaVariables.W;
	float * calculated = cudaVariables.gwtx_into_W;
	float * factor=cudaVariables.g_wtx;
	
	
		
	dim3 blockSize(16,16);
	dim3 gridSize((int)ceil(((float)p)/blockSize.x),(int)ceil(((float)n)/blockSize.y));
    
    multiplyColumnVise<<<gridSize, blockSize>>>(original,calculated,factor,n,n);
    cudaDeviceSynchronize();
    
	}
	
void subtractOnGPU(cudaVar cudaVariables,int n){
	float * gwtx_into_x1transpose_p=cudaVariables.g_wtx_X1_transpose;
	float * gwtx_into_W=cudaVariables.gwtx_into_W;
	float * target=cudaVariables.w_init;

	int blockSize, gridSize;
    blockSize = 512;
    gridSize = (int)ceil(((float)(n*n))/blockSize);

    subtractMatrices<<<gridSize, blockSize>>>(target,gwtx_into_x1transpose_p,gwtx_into_W,n);
    
    cudaDeviceSynchronize();

	}

	
void matrixMultiplyonGPU(float * d_A, float * d_B, float * d_C,int n,int p){
	
	gpu_blas_mmul(d_A, d_B, d_C, n, n, p);
	cudaDeviceSynchronize();

	}


void matrixMultiplyTransposeImprovedonGPU(cudaVar cudaVariables,float p_,int n,int p){
	float * d_A = cudaVariables.gwtx;
	float * d_B = cudaVariables.X1Transpose;
	float * d_C = cudaVariables.g_wtx_X1_transpose;
	
	gpu_blas_mmulImprove(d_A, d_B, d_C, n, p, n,p_);
	cudaDeviceSynchronize();
	
	}
	

