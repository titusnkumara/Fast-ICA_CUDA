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

int runSVDonCUDA(float * A,VectorXd& singularValue,MatrixXd& singularVectors,preprocessVariables* DevicePointers,int ROWS,int COLS){
	//initilaize matrix
	//int ROWS =1000;
	//int COLS =3;
	
	////MatrixXf DataMatrix = input.cast<float>();
	
    ////float* h_A = DataMatrix.data();
	
	
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
    printf("Performing singular value decomposition using CULA ... \n");

    time(&begin_time);
    status = culaDeviceSgesvd(jobu, jobvt, M, N, A, LDA, S, U, LDU, VT, LDVT);
    checkStatus(status);
	cout<<"Status of SVD= "<<status<<endl;
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
	DevicePointers->d_VT = VT;
	DevicePointers->d_VTT = VTT;
	DevicePointers->d_S = S;
	
	for (int i = 0; i < singularVectors.size(); i++){
		
		*(singularVectors.data() + i) = (double)VT_tmp[i];
	}
	
	for (int i = 0; i < singularValue.size(); i++){
		
		*(singularValue.data() + i) = (double)S_tmp[i];
	}
	
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
		output[index] = d_eigenVectors[index]*(1/sqrt(d_eigenValues[(n*x)+x]));
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
	
    devideByDiagonal<<<gridSize, blockSize>>>(d_eigenValues,d_eigenVectors,output,n,n);
	cudaDeviceSynchronize();
	
	MatrixXf h_eigenvectors(n,n);
	cudaMemcpy(h_eigenvectors.data(),output,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	cout<<"D_eigenvectors "<<endl<<h_eigenvectors<<endl;
	
	TransposeMatrixInCUBLAS(d_eigenVectors,d_eigenVectorT,n,n);
	gpu_blas_mmul(output,d_eigenVectorT,d_output_eigenVectorT,n,n,n);
	gpu_blas_mmul(d_output_eigenVectorT,d_w_init,output,n,n,n);
	
	return output;
	
}

float * memSetForSymDecorrelationCUDA(MatrixXf& w_init,float * d_w_init,float * d_VTT,int n){
	
	
	float * datapointer = w_init.data();
	cudaMalloc( (void**)&d_w_init, n*n*sizeof(float));checkCudaError();
	cudaMalloc( (void**)&d_VTT, n*n*sizeof(float));checkCudaError();
	
	cudaMemcpy( d_w_init, datapointer,n*n*sizeof(float), cudaMemcpyHostToDevice );checkCudaError();
	return d_w_init;
}



MatrixXd sym_decorrelation_cuda( float * d_VTT, float * d_w_init,int n){
	
	
	//float 
	float * d_w_init_tr;
	float * d_w_init_w_init_tr;
	
	//MatrixXf w_init_f = w_init.cast<float>();
	
	//cout<<"w_init_f"<<endl<<w_init_f<<endl;
	
	//float * datapointer = w_init_f.data();
	//cudaMalloc( (void**)&d_VTT, n*n*sizeof(float));checkCudaError();
	cudaMalloc( (void**)&d_w_init_tr, n*n*sizeof(float));checkCudaError();
	cudaMalloc( (void**)&d_w_init_w_init_tr, n*n*sizeof(float));checkCudaError();
	
	//save w_init to cuda
	//cudaMemcpy( d_w_init, datapointer,n*n*sizeof(float), cudaMemcpyHostToDevice );checkCudaError();
	
	TransposeMatrixInCUBLAS(d_w_init,d_w_init_tr,n,n);
	
	//MatrixXf w_init_f_tr(n,n);
	//cudaMemcpy(w_init_f_tr.data(),d_w_init_tr,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	
	//cout<<"w_init_f_tr"<<endl<<w_init_f_tr<<endl;
	
	gpu_blas_mmul(d_w_init,d_w_init_tr,d_w_init_w_init_tr,n,n,n);
	
	// MatrixXf w_init_f_tr_m(n,n);
	// cudaMemcpy(w_init_f_tr_m.data(),d_w_init_w_init_tr,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	
	// cout<<"W1 gpu"<<endl<<w_init_f_tr_m<<endl;
	
	
	//finding eigenvalues
	//The code to find SVD is follows, edit it to match relevant function
	

	culaStatus status;

	time_t begin_time;
    time_t end_time;
    int cula_time;

	
	
	//job parametes I am editing from here
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
	//cudaMalloc( (void**)&VR, LDVR*LDA*sizeof(float));checkCudaError();
  

	// Initialize CULA
    status = culaInitialize();
    
	 // Perform eigenvalue CULA 
    printf("eigenvalues using CULA ... \n");

    time(&begin_time);
	/* SGEEV prototype */
/*extern void sgeev( char* jobvl, char* jobvr, int* n, float* a,
                int* lda, float* wr, float* wi, float* vl, int* ldvl,
                float* vr, int* ldvr, float* work, int* lwork, int* info );*/
    status = culaDeviceSgeev(JOBVL,JOBVR, n, A, LDA, WR, WI,VL, LDVL,VR,LDVR);
    checkStatus(status);
	cout<<"Status of eigen= "<<status<<endl;
    time(&end_time);

	
    cula_time = (int)difftime( end_time, begin_time);
	
	
	
	culaShutdown();
	//copy back data,maybe should move before culashutdown

	
	MatrixXf Left(n,n);
	//MatrixXf Right(n,n);
	MatrixXf A_OUT(n,n);
	
	cudaMemcpy(Left.data(),VL,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	//cout<<"From CUDA W"<<endl<<devidedW<<endl;MatrixXf devidedW(n,n);
	//cudaMemcpy(Right.data(),VR,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(A_OUT.data(),A,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	cout<<"From CUDA Left"<<endl<<Left<<endl;
	//cout<<"From CUDA Right"<<endl<<Right<<endl;
	cout<<"From CUDA A"<<endl<<A_OUT<<endl;
	
	
	
	//invoking devide by diagonal
	float * output;
	output=invokeDevideByDiagonal(A,VL,d_w_init,n);
	d_VTT = output;
	MatrixXf h_output(n,n);
	cudaMemcpy(h_output.data(),output,n*n*sizeof(float),cudaMemcpyDeviceToHost);
	cout<<"CUDA sym_D output "<<endl<<h_output<<endl;
	
	MatrixXd outputD = h_output.cast<double>();
	//return outputD;
	
	
	
	cudaDeviceSynchronize();checkCudaError();
	//return EXIT_SUCCESS;
	
	return outputD;
	
	
	
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
	float * d_VT;
	float * d_Xf;
	float * d_X1;
	double * d_X;
	
	
	dim3 blockSizeNorm(16,16);
	dim3 gridSizeNorm((int)ceil(((float)p)/blockSizeNorm.x),(int)ceil(((float)n)/blockSizeNorm.y));
    
	
	d_VT = DevicePointers->d_VT;
	cudaMalloc((void**)&d_Xf, n*p*sizeof(float));checkCudaError();
	cudaMalloc((void**)&d_X1, n*p*sizeof(float));checkCudaError();
	//casting to float for SVD
	d_X =  DevicePointers->d_X;
	castToFloatInCUDA<<<gridSizeNorm, blockSizeNorm>>>(d_X,d_Xf,n,p);
	
	gpu_blas_mmul_X1(d_VT, d_Xf, d_X1,n,n,p);
	
	MatrixXf X1(n,p);
	cudaMemcpy(X1.data(),d_X1,n*p*sizeof(float),cudaMemcpyDeviceToHost);
	//cout<<"X1 cuda"<<endl<<X1<<endl;
	
}


void getMeanNormalizeOnCUDA(VectorXd& means, MatrixXd& X,int n,int p,preprocessVariables* DevicePointers ){
	
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
	cudaMemcpy( means.data(), d_means,  n*sizeof(double), cudaMemcpyDeviceToHost );checkCudaError();
	
	
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





cudaVar initializeCuda(preprocessVariables* DevicePointers,MatrixXd& X1,MatrixXd& w_init,cudaVar cudaVariables,int n,int p){
	
	//matrix sizes
	
	//MatrixXf f_W = W.cast<float>();
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
	//cudaMalloc( (void**)&cudaVariables.W, matsizeW );
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
	//float *dataFromW = f_W.data();
	float *dataFromW_init = f_w_init.data();
	
	
	
	//copy data to CUDA
	cudaMemcpy( cudaVariables.X1, dataFromX1, matsizeX1, cudaMemcpyHostToDevice );
	cudaMemcpy( cudaVariables.X1Transpose, datafromX1transpose, matsizeX1Transpose, cudaMemcpyHostToDevice );
	cudaVariables.W = DevicePointers->d_VTT;
	//cudaMemcpy( cudaVariables.W, dataFromW, matsizeW, cudaMemcpyHostToDevice );
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
	

