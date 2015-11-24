
#include <iostream>
#include "lib/helpers.cuh"

using namespace std;

int main(){

	float h_A[100];h_A[0]=123;h_A[1]=213;h_A[2]=3;

	int M= 10;
    int N = 10;
	
//	culaStatus status;
	
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
	
	cout<<"checkih_A"<<h_A[0]<<" "<<h_A[1]<<" "<<h_A[2]<<endl;
	cudaMalloc((void**)&A, M*N*sizeof(float ));checkCudaError();
	cudaMemcpy( A, h_A,  3*sizeof(float ), cudaMemcpyHostToDevice );checkCudaError();
	
	float * test = (float *)malloc(M*N*sizeof(float ));
	cudaMemcpy( test, A,  3*sizeof(float ), cudaMemcpyDeviceToHost );checkCudaError();
	
	//checking elements
	cout<<test[0]<<" "<<test[1]<<" "<<test[2]<<endl;

return 0;
}
