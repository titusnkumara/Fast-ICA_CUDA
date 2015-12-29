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

void checkStatus(culaStatus status)
{
    char buf[256];

    if(!status)
        return;

    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
    printf("%s\n", buf);

    culaShutdown();
    exit(EXIT_FAILURE);
}

using namespace Eigen;
using namespace std;
using Eigen::ArrayXXd ;
using Eigen::MatrixXd ;

#include "common.h"
#include <cublas_v2.h>
#include <iomanip>


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
void gpu_blas_mmul_X1(float *A, float *B, float *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const float alf = (float)sqrt((double)n);
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;
	
	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	cudaDeviceSynchronize();
	
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

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(double *A, double *B, double *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;
	
	// Do the actual multiplication
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

__global__
void memSetInCuda(double *d_singleArray,double num,int sizeofSingleArray){
	
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	if(x<sizeofSingleArray){
		
		d_singleArray[x] = num;
	}
	
}



/*
SVD functions


*/

int runSVDonCUDA(float * A,preprocessVariables* DevicePointers,int ROWS,int COLS){
	//initilaize matrix

	
	int M = ROWS;
    int N = COLS;
	
	culaStatus status;
	
	/* Setup SVD Parameters */
    int LDA = M;
    int LDU = M;
    int LDVT = N;
   
    ////float* A = NULL;
    float* S = NULL;
    float* U = NULL;
    float* VT = NULL;
    float* VTT = NULL;
	
	time_t begin_time;
    time_t end_time;
    int cula_time;

    char jobu = 'N';
    char jobvt = 'A';

	
	cudaMalloc( (void**)&S, imin(M,N)*sizeof(float));checkCudaError();
  
	cudaMalloc( (void**)&VT, LDVT*N*sizeof(float));checkCudaError();
	

	//malloc for d_VTT
	cudaMalloc( (void**)&VTT, LDVT*N*sizeof(float));checkCudaError();
	
	/* Initialize CULA */
    status = culaInitialize();
    
	 /* Perform singular value decomposition CULA */

    time(&begin_time);
    status = culaDeviceSgesvd(jobu, jobvt, M, N, A, LDA, S, U, LDU, VT, LDVT);
    checkStatus(status);
	cout<<"Status of SVD= "<<status<<endl;
    time(&end_time);

	
    cula_time = (int)difftime( end_time, begin_time);
	
	
	
	culaShutdown();

	DevicePointers->d_VT = VT;
	DevicePointers->d_VTT = VTT;
	DevicePointers->d_S = S;

	cudaDeviceSynchronize();checkCudaError();

    return EXIT_SUCCESS;

}

/**********************/
/* cuBLAS ERROR CHECK */
/**********************/
#ifndef cublasSafeCall
#define cublasSafeCall(err)     __cublasSafeCall(err, __FILE__, __LINE__)
#endif

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
    if( CUBLAS_STATUS_SUCCESS != err) {
        fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n",__FILE__, __LINE__,err); 
        assert(0); 
    }
}

void TransposeMatrixInCUBLAS(double *dv_ptr_in,double * dv_ptr_out,int m,int n){
	//m=number of rows
	//n = number of columns
	double alpha = 1.;
    double beta  = 0.;
    cublasSafeCall(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, dv_ptr_in, n, &beta, dv_ptr_in, n, dv_ptr_out, m)); 
    
}

void TransposeMatrixInCUBLAS(float *dv_ptr_in,float * dv_ptr_out,int m,int n){
	//m=number of rows
	//n = number of columns
	float alpha = 1.;
    float beta  = 0.;
    cublasSafeCall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, dv_ptr_in, n, &beta, dv_ptr_in, n, dv_ptr_out, m)); 
    
}


__global__
void normalizeInCUDA(double * d_X,double * d_means,int n,int p){

	//should optimize with shared memory
	int y = blockIdx.y*blockDim.y+threadIdx.y; //n
	int x = blockIdx.x*blockDim.x+threadIdx.x; //p
	
	
	if(y<n && x<p){
		int index;
		index = x*n+y;
		//printf("%d %d %lf %lf \n",index,index%n,d_X[index],d_means[index%n]);
		d_X[index] = d_X[index]- d_means[index%n];
		
	}	
}

__global__
void castToFloatInCUDA(double * input,float * output,int n,int p){

	//should optimize with shared memory
	int y = blockIdx.y*blockDim.y+threadIdx.y; //n
	int x = blockIdx.x*blockDim.x+threadIdx.x; //p
	
	
	if(y<n && x<p){
		int index;
		index = x*n+y;
		//printf("%d %d %lf %lf \n",index,index%n,d_X[index],d_means[index%n]);
		output[index] = (float)input[index];
		
	}	
}

__global__
void devideVTInCUDA(float *d_VT,float *d_S, int n){

	//should optimize with shared memory
	int y = blockIdx.y*blockDim.y+threadIdx.y; //n
	int x = blockIdx.x*blockDim.x+threadIdx.x; //p
	
	
	if(y<n && x<n){
		int index;
		index = x*n+y;
		//printf("%d %d %lf %lf \n",index,index%n,d_VT[index],d_S[index%n]);
		d_VT[index] = d_VT[index] / d_S[index%n];
		
	}	
}


__global__
void devideByDiagonal(float *d_eigenValues,float * d_eigenVectors,float * output,int n,int p){

	//should optimize with shared memory
	//n means row, p means columns
	//since this is square matrix, n=p
	int y = blockIdx.y*blockDim.y+threadIdx.y; //n
	int x = blockIdx.x*blockDim.x+threadIdx.x; //p
	
	
	if(y<n && x<p){
		int index;
		index = x*n+y;
		//printf("%d %d %d %f %d %f\n",x,y,index,d_eigenVectors[index], (n*x)+x, 1/sqrt(d_eigenValues[(n*x)+x]));
		float val = d_eigenValues[(n*x)+x];
		if(val<=0){
			printf("\nI Got negative\n");
			output = NULL;
			printf("Now error value %p\n",output);
		}
		output[index] = d_eigenVectors[index]*(1/sqrt(val));
	}	
}

float * invokeDevideByDiagonal(float * d_eigenValues, float * d_eigenVectors,float * d_w_init,int n){
	
	float * output;
	float * d_eigenVectorT;
	float * d_output_eigenVectorT;
	//get rid of this allocation that happens everytime
	cudaMalloc( (void**)&output, n*n*sizeof(float));checkCudaError();
	cudaMalloc( (void**)&d_eigenVectorT, n*n*sizeof(float));checkCudaError();
	cudaMalloc( (void**)&d_output_eigenVectorT, n*n*sizeof(float));checkCudaError();
	
	dim3 blockSize(16,16);
	dim3 gridSize((int)ceil(((float)n)/blockSize.x),(int)ceil(((float)n)/blockSize.y));
	
	
	// MatrixXf tmp(n,n);
	// cudaMemcpy(tmp.data(),d_eigenValues,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	// cout<<"Eigenvalues"<<endl<<tmp<<endl;
	// cudaMemcpy(tmp.data(),d_eigenVectors,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	// cout<<"EigenVectors"<<endl<<tmp<<endl;
	printf("Error value before %p\n",output);
    devideByDiagonal<<<gridSize, blockSize>>>(d_eigenValues,d_eigenVectors,output,n,n);
	cudaDeviceSynchronize();
	printf("Error value after %p\n",output);
	if(output==NULL){
		cout<<endl<<"returning null"<<endl;
		return NULL;
	}
	// cudaMemcpy(tmp.data(),output,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	// cout<<"output "<<endl<<tmp<<endl;
	
	
	TransposeMatrixInCUBLAS(d_eigenVectors,d_eigenVectorT,n,n);
	gpu_blas_mmul(output,d_eigenVectorT,d_output_eigenVectorT,n,n,n);
	gpu_blas_mmul(d_output_eigenVectorT,d_w_init,output,n,n,n);
	
	return output;
	
}

void memSetForSymDecorrelationCUDA(MatrixXf& w_init,preprocessVariables* DevicePointers,int n){
	
	
	float * datapointer = w_init.data();
	cudaMalloc( (void**)&(DevicePointers->d_w_init), n*n*sizeof(float));checkCudaError();
	cudaMalloc( (void**)&(DevicePointers->d_VTT), n*n*sizeof(float));checkCudaError();
	
	cudaMemcpy( DevicePointers->d_w_init, datapointer,n*n*sizeof(float), cudaMemcpyHostToDevice );checkCudaError();
	 
	//return d_w_init;
}



void sym_decorrelation_cuda(preprocessVariables* DevicePointers,int n){
	
	
	//float 
	float * d_w_init_tr;
	float * d_w_init_w_init_tr;

	// MatrixXf tmp(n,n);
	// cudaMemcpy(tmp.data(),DevicePointers->d_w_init,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	// cout<<"W_init before"<<endl<<tmp<<endl;
	
	cudaMalloc( (void**)&d_w_init_tr, n*n*sizeof(float));checkCudaError();
	cudaMalloc( (void**)&d_w_init_w_init_tr, n*n*sizeof(float));checkCudaError();

	TransposeMatrixInCUBLAS(DevicePointers->d_w_init,d_w_init_tr,n,n);
	
	gpu_blas_mmul(DevicePointers->d_w_init,d_w_init_tr,d_w_init_w_init_tr,n,n,n);
	
	
	//find eigenvalues
	culaStatus status;

	time_t begin_time;
    time_t end_time;
    int cula_time;

	
	
	//job parameters
    char JOBVL = 'V';
    char JOBVR = 'N';
	//int n is given
	float * A = d_w_init_w_init_tr;
	
	/* Setup Parameters leading dimension*/
    int LDA = n;
	int LDVL = n;
	int LDVR = n;
	
    float * WR = NULL;
	cudaMalloc( (void**)&WR, LDA*sizeof(float));checkCudaError();
  
	float * WI = NULL;
	cudaMalloc( (void**)&WI, LDA*sizeof(float));checkCudaError();
  
    float * W = NULL;
	cudaMalloc( (void**)&W, LDA*sizeof(float));checkCudaError();
  
	float * VL = NULL;
	cudaMalloc( (void**)&VL, LDVL*LDA*sizeof(float));checkCudaError();
  
	float * VR = NULL;

	// Initialize CULA
    status = culaInitialize();
    time(&begin_time);
	/* SGEEV prototype */
	/*extern void sgeev( char* jobvl, char* jobvr, int* n, float* a,
                int* lda, float* wr, float* wi, float* vl, int* ldvl,
                float* vr, int* ldvr, float* work, int* lwork, int* info );*/
    status = culaDeviceSgeev(JOBVL,JOBVR, n, A, LDA, WR, WI,VL, LDVL,VR,LDVR);
    checkStatus(status);
	cout<<"svd status "<<status<<endl;
    time(&end_time);
    cula_time = (int)difftime( end_time, begin_time);
	culaShutdown();
	
	// cudaMemcpy(tmp.data(),A,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	// cout<<"A after"<<endl<<tmp<<endl;
	
	// cudaMemcpy(tmp.data(),VL,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	// cout<<"VL after"<<endl<<tmp<<endl;
	// cudaMemcpy(tmp.data(),DevicePointers->d_w_init,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	// cout<<"d_w_init after"<<endl<<tmp<<endl;
	
	//invoking devide by diagonal
	float * output;
	output=invokeDevideByDiagonal(A,VL,DevicePointers->d_w_init,n);
	
	DevicePointers->d_VTT = output;	
}

void copyBackW1fromCUDA(MatrixXd& W1,float * d_W1,int n){
	
	MatrixXf tmp(n,n);
	cudaMemcpy(tmp.data(),d_W1,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	W1 = tmp.cast<double>();
}



void devideVTbySingularValues(float * d_VT,float * d_VTT, float * d_S,int n){
	dim3 blockSizeNorm(16,16);
	dim3 gridSizeNorm((int)ceil(((float)n)/blockSizeNorm.x),(int)ceil(((float)n)/blockSizeNorm.y));
	devideVTInCUDA<<<gridSizeNorm, blockSizeNorm>>>(d_VT,d_S,n);
	
	//Transpose d_VT to d_VTT
	TransposeMatrixInCUBLAS(d_VT,d_VTT,n,n);
	
	MatrixXf devidedW(n,n);
	cudaMemcpy(devidedW.data(),d_VTT,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	//cout<<"From CUDA W"<<endl<<devidedW<<endl;
	
}

void multiplyOnGPU_K_X(preprocessVariables* DevicePointers,int n,int p){
	//K=d_VT
	
	float * d_Xf;
	
	
	
	dim3 blockSizeNorm(16,16);
	dim3 gridSizeNorm((int)ceil(((float)p)/blockSizeNorm.x),(int)ceil(((float)n)/blockSizeNorm.y));
    
	
	cudaMalloc((void**)&d_Xf, n*p*sizeof(float));checkCudaError();
	cudaMalloc((void**)&(DevicePointers->d_X1), n*p*sizeof(float));checkCudaError();
	cudaMalloc((void**)&(DevicePointers->d_X1_T), p*n*sizeof(float));checkCudaError();
	//casting to float for SVD
	castToFloatInCUDA<<<gridSizeNorm, blockSizeNorm>>>(DevicePointers->d_X,d_Xf,n,p);
	
	gpu_blas_mmul_X1(DevicePointers->d_VT, d_Xf, DevicePointers->d_X1,n,n,p); //X1 matrix
	TransposeMatrixInCUBLAS(DevicePointers->d_X1,DevicePointers->d_X1_T,p,n); //X1 Transpose matrix

	
}

void copyKtoHost(MatrixXd& K,preprocessVariables* DevicePointers,int n){
	MatrixXf tmp(n,n);
	cudaMemcpy(tmp.data(),DevicePointers->d_VT,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	K = tmp.cast<double>();
}


void getMeanNormalizeOnCUDA(MatrixXd& X,int n,int p,preprocessVariables* DevicePointers ){
	
	double * d_means;
	double * d_X;
	double * d_singleArray;
	double * xpointer = X.data();

	double * d_X_tr; //device x transposed
	float * d_X_trf; //device x transposed float

	dim3 blockSizeNorm(16,16);
	dim3 gridSizeNorm((int)ceil(((float)p)/blockSizeNorm.x),(int)ceil(((float)n)/blockSizeNorm.y));
    
	cudaMalloc((void**)&d_X_tr, n*p*sizeof(double));checkCudaError();
	cudaMalloc((void**)&d_X_trf, n*p*sizeof(float));checkCudaError();
	
	cudaMalloc((void**)&d_means, n*sizeof(double));checkCudaError();
	cudaMalloc((void**)&d_X, n*p*sizeof(double));checkCudaError();
	cudaMalloc((void**)&d_singleArray, p*sizeof(double));checkCudaError();
	
	cudaMemcpy( d_X, xpointer,  n*p*sizeof(double), cudaMemcpyHostToDevice );checkCudaError();
	
	int blockSize = 512;
    int gridSize = (int)ceil(((float)(p))/blockSize);
	double memSetNumber = 1.0/p;
	//MemSet in CUDA
	memSetInCuda<<<gridSize, blockSize>>>(d_singleArray,memSetNumber,p*1*sizeof(double));
	gpu_blas_mmul(d_X, d_singleArray, d_means,n,p,1);
	//copy back data
	//cudaMemcpy( means.data(), d_means,  n*sizeof(double), cudaMemcpyDeviceToHost );checkCudaError();
	
	
	//normalizing
	//calling kernal
    normalizeInCUDA<<<gridSizeNorm, blockSizeNorm>>>(d_X,d_means,n,p);
    cudaDeviceSynchronize();
	//copy back data
	cudaMemcpy( X.data(), d_X,  n*p*sizeof(double), cudaMemcpyDeviceToHost );checkCudaError();
	
	
	
	//Transpose matrix using CUBLAS
	TransposeMatrixInCUBLAS(d_X,d_X_tr,p,n);
	/*
	//check correctness
	MatrixXd tmp(p,n);
	cudaMemcpy( tmp.data(), d_X_tr,  n*p*sizeof(double), cudaMemcpyDeviceToHost );checkCudaError();
	cout <<"X = "<<endl<<X<<endl;
	cout<<"Transposed X = "<<endl<<tmp<<endl;
	*/
	DevicePointers->d_X = d_X;
	
	//casting to float for SVD
	castToFloatInCUDA<<<gridSizeNorm, blockSizeNorm>>>(d_X_tr,d_X_trf,n,p);
    cudaDeviceSynchronize();
	/*
	//check correctness
	MatrixXf tmpf(p,n);
	cudaMemcpy( tmpf.data(), d_X_trf,  n*p*sizeof(float), cudaMemcpyDeviceToHost );checkCudaError();
	cout <<"X double = "<<endl<<X<<endl;
	cout<<"Transposed X float = "<<endl<<tmpf<<endl;
	*/
	DevicePointers->d_X_tr = d_X_tr;
	DevicePointers->d_X_trf = d_X_trf;
	

}





cudaVar initializeCuda(preprocessVariables* DevicePointers,cudaVar cudaVariables,int n,int p){
	
	//matrix sizes
	

	const int matsizeW = n*n*sizeof(float);
	const int matsizeW1 = n*n*sizeof(float);
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

	const int sizeofSingleArray = p*1*sizeof(float);
	const int sizeofComputedArray = n*1*sizeof(float);
	
	//malloc
	cudaMalloc( (void**)&cudaVariables.W1, matsizeW1 );
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
	cudaMalloc( (void**)&cudaVariables.d_singleArray, sizeofSingleArray );
	cudaMalloc( (void**)&cudaVariables.d_computeArray, sizeofComputedArray );
	cudaMallocHost((void **) &cudaVariables.hostpointer,matsizeW);
	
	
	//MemSet in CUDA
	int blockSize = 512;
    int gridSize = (int)ceil(((float)(p))/blockSize);
	memSetInCuda<<<gridSize, blockSize>>>(cudaVariables.d_singleArray,1.0,p*1*sizeof(float));

	
	
	//copy data to CUDA
	cudaVariables.X1 = DevicePointers->d_X1;
	cudaVariables.X1Transpose = DevicePointers->d_X1_T;
	cudaVariables.W = DevicePointers->d_VTT;
	cudaVariables.w_init = DevicePointers->d_w_init;

	

	cudaDeviceSynchronize();
	
	return cudaVariables;
	
	
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
	

