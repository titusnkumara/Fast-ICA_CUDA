
//#define RUNONCPU 1




#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>


#include <fcntl.h>    /* For O_RDWR */
#include <unistd.h>   /* For open(), creat() */
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <fstream>

#include <sys/time.h>
#include <time.h>


																				/*
																				 * definitions for cuda
																				 */
#include "common.h"



using namespace Eigen;
using namespace std;
using Eigen::ArrayXXd ;

#define _DEBUG 1
//#define _PRINTOUTPUT 0
#define ArgCount 5
#define PRECISION 10
#define MAX_ITER 500
#define TOL 0.000001 

//function to measure time
typedef unsigned long long timestamp_t;

static timestamp_t startTime,endTime;


																				/*
																				 * global pointers in CUDA Device
																				 *Saved
																				 */




struct size{
	int n;
	int p;
} dimensions;

struct results{
	MatrixXd S;
	MatrixXd W;
	int iterations;
} result;

void  readInputData(MatrixXd& X,char * file, int row,int column);
void getMean(VectorXd& means,MatrixXd& X,int n);
void normalize(MatrixXd& X,VectorXd& means,int rows);
void devide(MatrixXd& u,VectorXd& d,int cols);
MatrixXd generateRandomMatrix(int n);
void _ica_par(preprocessVariables* DevicePointers,MatrixXd& W,MatrixXd& w_init,int max_iter,double tol);
void _sym_decorrelation(MatrixXd& W,MatrixXd& w_init);
MatrixXd arrayMultiplierRowWise(MatrixXd u,ArrayXXd temp,int n);
ArrayXXd multiplyColumnWise(MatrixXd& g_wtx,MatrixXd& W,ArrayXXd& W_in,ArrayXXd& g_wtx_in);

void cube(MatrixXd& gwtx,MatrixXd& xin,ArrayXXd& x);
void cubed(MatrixXd& g_wtx,MatrixXd& xin,ArrayXXd& x);
void WriteResultToFile(MatrixXd& S,char * file);
void WriteTestToFile(VectorXd& V,char * file);
void WriteMatrixToFile(MatrixXd& S,char * file);
void printRowCols(MatrixXd& X);
void printTime(const char part[]);

	
	
																		/*
																		 * Helper functions
																		 * */
																				




void printMatrix(MatrixXd M){
	cout<<M.transpose()<<endl;
}


void printRowCols(MatrixXd& X){
	
	cout<<"<"<<X.rows()<<","<<X.cols()<<">"<<endl;
	}
	
void printRowCols(ArrayXXd& X){
	
	cout<<"<"<<X.rows()<<","<<X.cols()<<">"<<endl;
	}



void printTime(const char part[]){
	
	cout<< part <<" :"<<(endTime - startTime) / 1000000.0L<<endl;
	}
	
static timestamp_t get_timestamp (){
      struct timeval now;
      gettimeofday (&now, NULL);
      return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
      }	
	

																				/*
																				 * CUDA functions
																				 * */
#include "mycuda.h"

																				/*
																				FastICA class
																				*/
class FastICA{
	
	private:
	
	int n_components;
	int max_iter;
	double tol;
	/*
	//this called by fit_transform
	MatrixXd _fit(MatrixXd X){
		
		//call fastica function
		return fastica(X,n_components,max_iter,tol);
	}
	*/
	public:
	
	FastICA(int numOfComponents){
		
		//initializing some variables
		n_components = numOfComponents;
		max_iter = MAX_ITER;
		tol = TOL;
	}
	
	void fit_transform(MatrixXd& X, MatrixXd& _S, MatrixXd& W){
		fastica(X,n_components,_S,W,max_iter,tol);
	}
	
	void fastica(MatrixXd& X,int n_components, MatrixXd& S, MatrixXd& W,int max_iter, double tol);
	
	
};



																		/*
																		The fastica function
																		Preprocessing and calling ICA parallel function
																		*/

void FastICA::fastica(MatrixXd& X,int n_components, MatrixXd& S, MatrixXd& W,int max_iter, double tol){
	//n=rows,p=columns
	int n,p;
	
	create_blas_handler();	//creating blas handler for matrix multiplications
	
	//take dimensions from global structure
	n = dimensions.n;
	p = dimensions.p;
	
	//timestamp structure
	timestamp_t prepr0 = get_timestamp();
	
	
	VectorXd means(n);
	MatrixXd u(n,n);	//u of svd	
	VectorXd d;	//d of svd
	VectorXd singularValue(n,1);
	MatrixXd singularVectors(n,n);
	MatrixXd K(n,n);
	MatrixXd X1(n,p);
	MatrixXd w_init(n,n);	//random matrix
	MatrixXd unmixedSignal;	//unmixing X using W
	
	
	
																				/*
																				 * Here is the preprocessing part
																				 * */
																				
	
	//computing mean for every row
	#ifdef _DEBUG
	startTime= get_timestamp();
	#endif
	//run on GPU
	preprocessVariables DevicePointers; //this structure holds variable in Device memory in structure
	getMeanNormalizeOnCUDA(X,n,p,&DevicePointers);
	#ifdef _DEBUG
    endTime = get_timestamp();
    printTime("Mean and Normalize function");
	#endif

	//CUDASVD using CULA library
	#ifdef _DEBUG
	startTime = get_timestamp();
	#endif

	//CUDASVD will run
	runSVDonCUDA(DevicePointers.d_X_trf,&DevicePointers,p,n);
	
	#ifdef _DEBUG
	endTime = get_timestamp();
    printTime("SVD");
	#endif

	
	devideVTbySingularValues(DevicePointers.d_VT,DevicePointers.d_VTT,DevicePointers.d_S,n);//gpu
	multiplyOnGPU_K_X(&DevicePointers,n,p);//gpu
	
	
	
	
	w_init = generateRandomMatrix(n);
	
	//measure finished time
    timestamp_t prepr1 = get_timestamp();
    cout<<"preprocess "<<(prepr1 - prepr0) / 1000000.0L<<endl;
	
	
	/*
	 * Preprocessing is finished
	 * */

																				/*
																				 * ICA parallel loop function is calling
																				 * */

	
	

	//calling ica parallel function
	timestamp_t ica0 = get_timestamp();
	_ica_par(&DevicePointers,W,w_init,max_iter,tol); //now we have mixed matrix W
    timestamp_t ica1 = get_timestamp();
	
	cout<<"ICA_Parallel_Loop: "<<(ica1 - ica0) / 1000000.0L<<endl;
	
																				/*
																				 * Post calculation starts here
																				 * */
	
	startTime = get_timestamp();
	copyKtoHost(K,&DevicePointers,n); 	//Copy K to CPU
	unmixedSignal = (W*K)*X;
	S = unmixedSignal.transpose();
	result.S = S;
    endTime = get_timestamp();
    printTime("PostCalculations");
	
}

																				/*
																				Writing S into file
																				*/
void WriteResultToFile(MatrixXd& S,char * file){	
	FILE * fp = fopen(file,"w");
	int  i,j;
	int row = dimensions.n;
	int column = dimensions.p;
	//printf("%d %d %ld %ld\n",row,column,S.rows(),S.cols());
  for(i=0;i<row;i++){
	  for(j=0;j<column;j++){
		  fprintf(fp,"%lf ",S(j,i));
	  }
	  fprintf(fp,"\n");	  
  }
  
	
}

void WriteMatrixToFile(MatrixXd& S,char * file){	
	FILE * fp = fopen(file,"w");
	int  i,j;
	int row = dimensions.n;
	int column = dimensions.n;
	//printf("%d %d %ld %ld\n",row,column,S.rows(),S.cols());
  for(i=0;i<row;i++){
	  for(j=0;j<column;j++){
		  fprintf(fp,"%lf ",S(j,i));
	  }
	  fprintf(fp,"\n");	  
  }
  
	
}

void WriteTestToFile(VectorXd& V,char * file){	
	FILE * fp = fopen(file,"w");
	int  j;
	int column = dimensions.n;
	
	  for(j=0;j<column;j++){
		  fprintf(fp,"%lf ",V(j));
	  }
	  fprintf(fp,"\n");	  
  
  
	
}


void _sym_decorrelation(MatrixXd& W1,MatrixXd& w_init){
	
	MatrixXd wt;
	int n = dimensions.n;
	int i,j;
	
	MatrixXd s(1,n);	//eigenvalues
	MatrixXd u(n,n);	//eigenvectors
	MatrixXcd values;	//complex array returned by eigenvalues
	MatrixXcd vectors;	//complex array returned by eigenvectors
	
	wt = w_init.transpose();
	W1	= w_init * wt;
	

	
	//My Eigen value function call
	MatrixXd eigenValues(n,1);
	MatrixXd eigenVectors(n,n);

	
	EigenSolver<MatrixXd> eigenSolver(W1,true);	//initializing eigen solver
	
	values = eigenSolver.eigenvalues();
	for(i=0;i<n;i++){
		s(0,n-i-1)= values(i,0).real();
	}
	
	vectors = eigenSolver.eigenvectors();
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			u(i,j) = vectors(i,n-j-1).real();
		}
	}

	W1 = (arrayMultiplierRowWise(u,(1/sqrt(s.array())),n) * u.transpose())*w_init;

}


/*
Multiply each row of u by temp
*/
MatrixXd arrayMultiplierRowWise(MatrixXd u,ArrayXXd temp,int n){
	ArrayXXd uArray = u.array();
	int i;
	//cout<<"Eigenvalues CPU * "<<endl<<temp<<endl;
	for(i=0;i<n;i++){
		//cout<<"eigenvector CPU "<<endl<<uArray.row(i)<<endl;
		
		uArray.row(i) *= temp;
	}
	return uArray.matrix();
}





																				/*
																				Parallel ICA algorithm
																				*/

void _ica_par(preprocessVariables* DevicePointers,MatrixXd& W,MatrixXd& w_init,int max_iter,double tol){
	

	double p_;	//number of samples
	int i;
	
	int n = dimensions.n;
	int p = dimensions.p;
	p_ = (double)dimensions.p;
	
	
	MatrixXd W1(n,n);
	double lim;	//limit to check with tolerance
	float limFromCuda;
	bool success = false;
	
	//w_init is random matrix
	//W is d_VTT
	//Copy winit to float variable
	
	MatrixXf w_init_f(n,n);
	w_init_f = w_init.cast<float>();
	memSetForSymDecorrelationCUDA(w_init_f,DevicePointers,dimensions.n);
	

	sym_decorrelation_cuda(DevicePointers,n);

	

	//ica main loop

	MatrixXf tmp(n,n);
	cudaVar cudaVariables;
	cudaVariables = initializeCuda(DevicePointers,cudaVariables,n,p);

	timestamp_t loopStartTime = get_timestamp();
	
	//timer variables
	double matmultiplicationTime=0;
	double cubeTime = 0;
	double transposeMulTime = 0;
	double columnviseTime = 0;
	double subtractTime = 0;
	double copyTime = 0;
	double symDecorelationTime = 0;
	double limTime = 0;
	double saveWTime = 0;
	
	cout<<"Cuda initialization done"<<endl;
	for(i=0;i<max_iter;i++){
	cout <<endl<<i<<endl;
	cout<<"mul 0"<<endl;
	startTime = get_timestamp();
		matrixMultiplyonGPU(cudaVariables.W,cudaVariables.X1,cudaVariables.product,dimensions.n,dimensions.p);	//dot product in gpu
	endTime = get_timestamp();
	matmultiplicationTime+=(endTime - startTime) / 1000000.0L;
	cout<<"mul 1"<<endl;
	
	cout<<"cube 0"<<endl;
	startTime = get_timestamp();
		cubeOnGPU(cudaVariables,dimensions.n,dimensions.p);									//find g,g' in gpu
	endTime = get_timestamp();
	cubeTime+=(endTime - startTime) / 1000000.0L;
	cout<<"cube 1"<<endl;
	
	cout<<"matrix 0"<<endl;
	startTime = get_timestamp();
		matrixMultiplyTransposeImprovedonGPU(cudaVariables,p_,dimensions.n,dimensions.p);				//matrix multiplication on GPU																
	endTime = get_timestamp();
	transposeMulTime+=(endTime - startTime) / 1000000.0L;	
	cout<<"matrix 1"<<endl;
		
	cout<<"col 0"<<endl;
	startTime = get_timestamp();
		multiplyColumnViseOnGPU(cudaVariables,dimensions.n,dimensions.p);						//Done  gwtx_into_W in CUDA
	endTime = get_timestamp();
	columnviseTime+=(endTime - startTime) / 1000000.0L;		
	cout<<"col 1"<<endl;
	
	cout<<"col 0"<<endl;
	startTime = get_timestamp();
		subtractOnGPU(cudaVariables,dimensions.n);								//subtraction on gpu
	endTime = get_timestamp();
	subtractTime+=(endTime - startTime) / 1000000.0L;
	cout<<"col 1"<<endl;
	
	//setup for symmetric decorrelation
	DevicePointers->d_w_init = cudaVariables.w_init;
	
	cout<<"sym 0"<<endl;
	startTime = get_timestamp();
		sym_decorrelation_cuda(DevicePointers,n);
		//check for negative eigenValues
		if(DevicePointers->d_VTT==NULL){
			cout<<"Negative EigenValues"<<endl;
			break;
		}
	endTime = get_timestamp();
	symDecorelationTime+=(endTime - startTime) / 1000000.0L;
	cout<<"sym 1"<<endl;
	//Then I change into host machine code
	//copy w1 from cuda
	cout<<"copy 0"<<endl;
	startTime = get_timestamp();
		copyBackW1fromCUDA(W1,DevicePointers->d_VTT,n);
		//cout<<"W1"<<endl<<W1<<endl<<i<<endl;
	endTime = get_timestamp();
	copyTime+=(endTime - startTime) / 1000000.0L;	
	cout<<"copy 1"<<endl;
	
	startTime = get_timestamp();

	
	cout<<"lim 0"<<endl;
	//limFromCuda = limFunctionOnCuda(cudaVariables.W1,cudaVariables.W,cudaVariables.W1intoWT,cudaVariables.diagonal,cudaVariables.answer);
	lim =  ((((((W1*W.transpose()).diagonal()).array()).abs()) - 1).abs()).maxCoeff();	//max(abs(abs(diag(dot(W1, W.T))) - 1))
	endTime = get_timestamp();
	limTime+=(endTime - startTime) / 1000000.0L;
	cout<<"lim 1"<<endl;
	W = W1;							//keep W as W1 to next loop
	cudaVariables.W = DevicePointers->d_VTT;

	if(lim<tol){
		success = true;
		break;
		}
		
		
	}
	cout<<"loop done"<<endl;
	//destroy blas handler
	destroy_blas_handler();
	
	
	//timing of whole loop
	timestamp_t loopEndTime = get_timestamp();
	
	cout<<"matmultiplicationTime: "<<matmultiplicationTime<<endl;;
	cout<<"cubeTime: "<<cubeTime<<endl;
	cout<<"transposeMulTime: "<<transposeMulTime<<endl;
	cout<<"columnviseTime: "<<columnviseTime<<endl;
	cout<<"subtractTime: "<<subtractTime<<endl;
	cout<<"copyTime: "<<copyTime<<endl;
	cout<<"symDecorelationTime: "<<symDecorelationTime<<endl;
	cout<<"limTime: "<<limTime<<endl;
	cout<<"saveWTime: "<<saveWTime<<endl;
	
	cout<<"Total_Above: "<<matmultiplicationTime+cubeTime+transposeMulTime+columnviseTime+subtractTime+copyTime+symDecorelationTime+limTime<<endl;
	cout<< "Main_loop"<<" :"<<(loopEndTime - loopStartTime) / 1000000.0L<<endl;
	cout<<"Iterations: "<<i<<endl;
	cout<<"Time_per_iteration: "<<((loopEndTime - loopStartTime) / 1000000.0L)/i<<endl;
	
	
	result.iterations = i+1;	//save iterations
	if(!success){
		cout<<"!!!!! did not converged, increase the max_iter count!!!!!"<<endl;
	}
}

ArrayXXd multiplyColumnWise(MatrixXd& g_wtx,MatrixXd& W,ArrayXXd& W_in,ArrayXXd& g_wtx_in){
	W_in = W;
	g_wtx_in =  g_wtx;
	//cout<<"sizes win "<<W_in.rows()<<" "<<W_in.cols()<<endl;
	//cout<<"sizes "<<g_wtx_in.rows()<<" "<<g_wtx_in.cols()<<endl;
	int n = W_in.cols();
	int i;
	for(i=0;i<n;i++){
		W_in.col(i)*=g_wtx_in;
	}
	return W_in;
}
/*
void cube(MatrixXd& gwtx,MatrixXd& xin,ArrayXXd& x){
	//cout<<xin.rows()<<" "<<xin.cols();
	x = xin.array();	//convert to Array
	x*=(x*x);
	gwtx = x.matrix();
}

void cubed(MatrixXd& g_wtx,MatrixXd& xin,ArrayXXd& x){
	x = xin.array();	//convert to Array
	x*=x;
	xin =(3*x).matrix();	//3*x^2
	

	//finding sum
	
	int i;
	double sum=0;
	//printf("Second row of cubed function cpu");
	for(i=0;i<dimensions.p;i++){
		sum+=xin(1,i);
		//printf("%f ",xin(1,i));
		}
	
	//cout<<"Sum from CPU is for second row"<<sum<<endl;
	
	MatrixXd means(dimensions.n,1);	//mean of each row
	getMean(means,xin,dimensions.n);
	
	g_wtx = means;
}
*/

//generate random matrix
//for convinience I put the matrix same as python solution

MatrixXd generateRandomMatrix(int n){
/*
	MatrixXd x(3,3);
	x << 1, 2, 2,
	3, 2, 1,
	2, 1, 0;
	cout<<x.rows()<<" "<<x.cols()<<endl;
	return x;*/
	return MatrixXd::Random(n,n);
}


void devide(MatrixXd& u,VectorXd& d,int cols){
	//each column of u should devide by each row of d
	int i;
	for(i=0;i<cols;i++){
		u.col(i) /= d(i,0);
	}
}

void normalize(MatrixXd& X,VectorXd& means,int rows){
	
	//do element vise operation for every element
	//convert it to array and do the task
	int i;
	for(i=0;i<rows;i++){
		X.row(i) = X.row(i).array() - means(i);
	}
	
}

void getMean(VectorXd& means,MatrixXd& X,int n){
	int i;
	for(i=0;i<n;i++){
		means(i) = X.row(i).mean();
		
	}
}





int main( int argc, char *argv[])
{
	
	#ifdef _DEBUG
	timestamp_t  main0 = get_timestamp();
	#endif
	
	
	//standard out precision
	cout.precision(PRECISION);
	
	int row,column;
	
	
	
	
	if ( argc != ArgCount ){ 
		// We print argv[0] assuming it is the program name
		// Should provide row and column with the program name
		cout<<"usage: "<< argv[0] <<" <row> <column> <inputfile> <outputfile>\n";
		return 0;
	}
	
	//Take rows and columns from command line arguments
	//We take this much of data from input
	row=atoi(argv[1]);
    column=atoi(argv[2]);
	
	cout<<"dataset "<<row<<" "<<column<<endl;
	
	
	//Now we can create Observation matrix
	//Input data read into this matrix
	MatrixXd X(row,column);
	//cout<<"X ";printRowCols(X);
	//Result
	MatrixXd _S(column,row);
	//cout<<"_S ";printRowCols(_S);
	
	MatrixXd W(row,row);
	//cout<<"W ";printRowCols(W);
	
	
	
	//set global dimension structure
	//So we don't have to compute size of array each time
	dimensions.n = row;
	dimensions.p = column;
	
	//reading
	startTime = get_timestamp();
	readInputData(X,argv[3],row,column);
    endTime = get_timestamp();
    printTime("Reading");
    
	
	
	//fastica
	FastICA ica = FastICA(row);
	ica.fit_transform(X,_S,W); //X input, _S separated, W weight matrix
	
	#ifdef _DEBUG
	printf("\n");
	timestamp_t  main1 = get_timestamp();
    cout<<"TOTAL_TIME_FOR_ICA "<<(main1 - main0) / 1000000.0L<<endl;
	#endif
	
	timestamp_t write0 = get_timestamp();
	WriteResultToFile(result.S,argv[4]);
	timestamp_t write1 = get_timestamp();
    cout<<"writing "<<(write1 - write0) / 1000000.0L<<endl;
	
	
	return 0;	
}


/*
Function to read input data and put them into the matrix
arguments
int row - number of rows to scan
int column - number of columns to read

*/
void  readInputData(MatrixXd& X,char * file, int row,int column){
	
	FILE * fp = fopen(file,"r");
	int  i,j,r;
	double temp;
  
  for(i=0;i<row;i++){
	  for(j=0;j<column;j++){
		  r=fscanf(fp,"%lf",&temp);
		  X(i,j) = temp;
	  }
	  while(fgetc(fp)!='\n');
	  
  }
  
}
	
