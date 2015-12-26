
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
void _ica_par(preprocessVariables* DevicePointers,MatrixXd& W,MatrixXd& X1,MatrixXd& w_init,int max_iter,double tol);
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
																		*/

void FastICA::fastica(MatrixXd& X,int n_components, MatrixXd& S, MatrixXd& W,int max_iter, double tol){
	//n=rows,p=columns
	int n,p;
	create_blas_handler();	//creating blas handler for matrix multiplications
	
	//take dimensions from global structure
	n = dimensions.n;
	p = dimensions.p;
	
	
	
	
	timestamp_t prepr0 = get_timestamp();
	
	
	/*
	MatrixXd means(n,1);	//mean of each row
	cout<<"means "<<endl;
	printRowCols(means);
	*/
	VectorXd means(n);
	MatrixXd u(n,n);	//u of svd
	//cout<<"u ";printRowCols(u);
	
	VectorXd d;	//d of svd
	VectorXd singularValue(n,1);
	MatrixXd singularVectors(n,n);
	
	MatrixXd K(n,n);
	//cout<<"K ";	printRowCols(K);
	
	MatrixXd X1(n,p);
	//cout<<"X1 ";printRowCols(X1);
	
	MatrixXd w_init(n,n);	//random matrix
	//cout<<"w_init ";printRowCols(w_init);
	
	MatrixXd unmixedSignal;	//unmixing X using W
	
	
																				/*
																				 * Here is the preprocessing part
																				 * */
																				
	
	//computing mean for every row
	#ifdef _DEBUG
	startTime= get_timestamp();
	#endif
	
	//this matrix should removed
	//MatrixXd X_outnorm(n,p);
	#ifdef RUNONCPU
	getMean(means,X,n);
	normalize(X,means,n);
	//reusing S as tr
	S = X.transpose(); //***************************************/ This must avoid
	#else
	//run on GPU
	preprocessVariables DevicePointers;
	getMeanNormalizeOnCUDA(means,X,n,p,&DevicePointers);
	//cout<<"means from cuda = "<<endl<<means<<endl;
	#endif

	#ifdef _DEBUG
    endTime = get_timestamp();
    printTime("Mean and Normalize function");
	#endif

	//SVD
	#ifdef _DEBUG
	startTime = get_timestamp();
	#endif

	#ifdef RUNONCPU
	//if this is defined this will run on CPU
	JacobiSVD<MatrixXd> svd(S, ComputeThinV);
	//reusing W as u JacobiSVD
	W =svd.matrixV();
	d = svd.singularValues();
	//cout<<"W"<<endl<<W<<endl;
	//cout<<"d"<<endl<<d<<endl;
	
	//WriteTestToFile(d,"cpusingular.txt");
	//WriteMatrixToFile(W,"VTcpu.txt");
	
	#else
	//CUDASVD will run
	runSVDonCUDA(DevicePointers.d_X_trf,singularValue,singularVectors,&DevicePointers,p,n);
	W = singularVectors.transpose();
	d = singularValue;
	//WriteTestToFile(d,"gpusingular.txt");
	//WriteMatrixToFile(W,"VTgpu.txt");
	#endif
	
	

	#ifdef _DEBUG
	endTime = get_timestamp();
    printTime("SVD");
	#endif

	
	
	
	//cout<<"W before devide"<<endl<<W<<endl;
	devide(W,d,n);//cpu
	//cout<<"d"<<endl<<d<<endl;
	//cout<<"W after devide"<<endl<<W<<endl;
	K = W.transpose();//cpu
	//cout<<"K from CPU"<<endl<<K<<endl;
	
	//cout<<"(K*X)*sqrt(p)"<<endl<<(K*X)*sqrt(p)<<endl;
	//x1 cannot replaced since it uses X also
	X1 = (K*X)*sqrt(p);//cpu
	//cout<<"X ";printRowCols(X);
	//cout<<"X1 ";printRowCols(X1);
	
	#ifndef RUNONCPU
	//multiply in GPU
	//K is already in GPU
	devideVTbySingularValues(DevicePointers.d_VT,DevicePointers.d_VTT,DevicePointers.d_S,n);//gpu
	multiplyOnGPU_K_X(&DevicePointers,n,p);//gpu
	#endif
	
	
	#ifdef _PRINTOUTPUT
	cout<<"final preprocess "<<endl;
	cout<<X1.transpose()<<endl;
	#endif
	
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

	
	timestamp_t ica0 = get_timestamp();
	//calling the _ica_par paralleld ica algorithm function
	//it will return W
	
	cout<<"Printing current matrices that passes to _ica_par"<<endl;
	//cout<<"W"<<endl<<W<<endl;
	//cout<<"X1 cpu"<<endl<<X1<<endl;
	
	_ica_par(&DevicePointers,W,X1,w_init,max_iter,tol);
	//now we have mixed matrix W
	
	//measure finished time
    timestamp_t ica1 = get_timestamp();
	
	cout<<"ICA_Parallel_Loop: "<<(ica1 - ica0) / 1000000.0L<<endl;
	
																				/*
																				 * Post calculation starts here
																				 * */
	
	startTime = get_timestamp();
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
	
	//cout<<"W1 size"<<endl;
	//cout<<W1.rows()<<" "<<W1.cols();
	wt = w_init.transpose();
	W1	= w_init * wt;
	
	cout<<"W1 cpu"<<endl<<W1<<endl;
	/*
	cout<<"w_init cpu"<<endl;
	cout<<w_init<<endl;
	*/
	
	/*
	Since Eigenvalue compute give complex structure
	We should parse it into MatrixXd type
	*/
	
	//My Eigen value function call
	MatrixXd eigenValues(n,1);
	MatrixXd eigenVectors(n,n);
	
	
	/*
	cout<<"Eigen values/vectors for"<<endl;
	
	cout<<W<<endl;
	*/
	
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
	
	
	
	cout<<"Native eigen values"<<endl;
	cout<<s<<endl;
	cout<<"Native eigen vectors"<<endl;
	cout<<u<<endl;
	
	/*
	cout<<"(1/sqrt(s.array()))"<<endl;
	cout<<(1/sqrt(s.array()))<<endl;
	cout<<"u"<<endl;
	cout<<u<<endl;
	cout<<"arrayMultiplierRowWise(u,(1/sqrt(s.array())),n)"<<endl;
	cout<<arrayMultiplierRowWise(u,(1/sqrt(s.array())),n)<<endl;
	cout<<"cpu u.transpose()"<<endl;
	cout<<u.transpose()<<endl;
	cout<<"(arrayMultiplierRowWise(u,(1/sqrt(s.array())),n) * u.transpose()) in cpu"<<endl;
	cout<<(arrayMultiplierRowWise(u,(1/sqrt(s.array())),n) * u.transpose())<<endl;
	*/
	cout<<"CPU ans"<<endl<<arrayMultiplierRowWise(u,(1/sqrt(s.array())),n)<<endl;
	
	W1 = (arrayMultiplierRowWise(u,(1/sqrt(s.array())),n) * u.transpose())*w_init;
	
	/*
	cout<<"final W1 in cpu"<<endl;
	cout<<W1<<endl;
	*/
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

void _ica_par(preprocessVariables* DevicePointers,MatrixXd& W,MatrixXd& X1,MatrixXd& w_init,int max_iter,double tol){
	
	
	//What I recieve
	//cout<<endl<<"ica_par"<<endl;
	//cout<<"W ";printRowCols(W);
	//cout<<"X1 ";printRowCols(X1);
	//cout<<"w_init ";printRowCols(w_init);
	//cout<<"S ";printRowCols(S);
	
	//we want symmetrical corellation of w_init
	double p_;	//number of samples
	int i;
	//ArrayXXd gwtx_into_x1transpose_p;
	//ArrayXXd gwtx_into_W;
	
	//ArrayXXd x(dimensions.n,dimensions.p);
	
	int n = dimensions.n;
	int p = dimensions.p;
	
	MatrixXd W1(dimensions.n,dimensions.n);
	double lim;	//limit to check with tolerance
	float limFromCuda;
	bool success = false;
	
	//w_init is random matrix
	//W is d_VTT
	
	//MatrixXd W2(dimensions.n,dimensions.n);
	
	
	MatrixXf w_init_f(n,n);
	w_init_f = w_init.cast<float>();
	memSetForSymDecorrelationCUDA(w_init_f,DevicePointers->d_w_init,dimensions.n);
	
	cout<<"w_init before"<<endl<<w_init_f<<endl;
	W=sym_decorrelation_cuda(DevicePointers->d_w_init,DevicePointers->d_VTT,dimensions.n);
	//_sym_decorrelation(W,w_init);
	//cout<<"W GPU"<<endl<<W<<endl;
	//cout<<"W1 CPU"<<endl<<W<<endl;
	
	
	
	p_ = (double)dimensions.p;
	
	//MatrixXd gwtx(dimensions.n,dimensions.p);
	//MatrixXd g_wtx(dimensions.n,1);
	//ArrayXXd W_in(dimensions.n,dimensions.n);
	//ArrayXXd g_wtx_in(dimensions.n,1);
	//ica main loop
	
	
																				/*
																				 * ToDO:
																				 * Pass the required variables to CUDA
																				 * Call INIT function that allocate memory inside cuda
																				 * I should pass data from host:  
																				 * X1:Preprocessed Data matrix
																				 * W:Sym_decorrelation results
																				 * w_init:random matrix
																				 * 
																				 * 
																				 * I should allocate memory in cuda for:
																				 * prod:result of dot product
																				 * gwtx:results from g()
																				 * g_wtx:results form g'()
																				 * 
																				 */


	MatrixXf tmp(dimensions.n,dimensions.n);
	
	cudaVar cudaVariables;
	
	cout<<"w_init after"<<endl<<w_init_f<<endl;
	cudaVariables = initializeCuda(DevicePointers,W,X1,w_init,cudaVariables,dimensions.n,dimensions.p);
	
	
	
	
	
	//testing eigen function
	//test01();
	
	/*
	cout<<"W="<<endl<<W<<endl;
	cout<<"X1="<<endl<<X1<<endl;
	*/
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
	
	
	for(i=0;i<max_iter;i++){

	startTime = get_timestamp();
		matrixMultiplyonGPU(cudaVariables.W,cudaVariables.X1,cudaVariables.product,dimensions.n,dimensions.p);	//dot product in gpu
	endTime = get_timestamp();
	matmultiplicationTime+=(endTime - startTime) / 1000000.0L;
	
	startTime = get_timestamp();
		cubeOnGPU(cudaVariables,dimensions.n,dimensions.p);									//find g,g' in gpu
	endTime = get_timestamp();
	cubeTime+=(endTime - startTime) / 1000000.0L;
	
	
	startTime = get_timestamp();
		matrixMultiplyTransposeImprovedonGPU(cudaVariables,p_,dimensions.n,dimensions.p);				//matrix multiplication on GPU																
	endTime = get_timestamp();
	transposeMulTime+=(endTime - startTime) / 1000000.0L;	
		
		
	startTime = get_timestamp();
		multiplyColumnViseOnGPU(cudaVariables,dimensions.n,dimensions.p);						//Done  gwtx_into_W in CUDA
	endTime = get_timestamp();
	columnviseTime+=(endTime - startTime) / 1000000.0L;		
		
	startTime = get_timestamp();
		subtractOnGPU(cudaVariables,dimensions.n);								//subtraction on gpu
	endTime = get_timestamp();
	subtractTime+=(endTime - startTime) / 1000000.0L;
		
	
	startTime = get_timestamp();
		//Then I have w_init in gpu, I should copy it back
		//I have reused gwtx_into_x1transpose_p memory to save subtracted answer also
		copyBackW_initfromCUDA(w_init,tmp,cudaVariables.w_init,cudaVariables.hostpointer,cudaVariables.tmp_w_init,dimensions.n);
		//cout<<"w_init ";printRowCols(w_init);
	endTime = get_timestamp();
	copyTime+=(endTime - startTime) / 1000000.0L;	
		
		
	
		
		
		//Then I change into host machine code
	
	startTime = get_timestamp();
		_sym_decorrelation(W1,w_init);
	endTime = get_timestamp();
	symDecorelationTime+=(endTime - startTime) / 1000000.0L;
	
	//calling symDecorelation in cuda
	//cout<<"Calling _sym_decorrelationOnCuda loop"<<endl;
	//cudaVariables = _sym_decorrelationOnCuda(cudaVariables);
	
	
	startTime = get_timestamp();
	/*
	cout<<"W1 * W.transpose() in cpu"<<endl;
	cout<<W1*W.transpose()<<endl;
	
	cout<<"((((((W1*W.transpose()).diagonal()).array()).abs()) - 1).abs()) in cpu"<<endl;
	cout<<(((((W1*W.transpose()).diagonal()).array()).abs()) - 1).abs()<<endl;
	*/
	
	//limFromCuda = limFunctionOnCuda(cudaVariables.W1,cudaVariables.W,cudaVariables.W1intoWT,cudaVariables.diagonal,cudaVariables.answer);
	lim =  ((((((W1*W.transpose()).diagonal()).array()).abs()) - 1).abs()).maxCoeff();	//max(abs(abs(diag(dot(W1, W.T))) - 1))
	endTime = get_timestamp();
	limTime+=(endTime - startTime) / 1000000.0L;
	
	
	startTime = get_timestamp();
		W = W1;							//keep W as W1 to next loop
		saveW1inGPU(W1,cudaVariables,dimensions.n); //CUDA save W1 to W
	endTime = get_timestamp();
	saveWTime+=(endTime - startTime) / 1000000.0L;
		/*
		cout<<"W "<<endl;
		cout<<W<<endl;
		*/
		//cout<<"iter: "<<i<<" lim: "<<lim<<"\n";
		
	
	
		if(lim<tol){
			success = true;
			break;
		}
		
		
	}
	
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
	
	cout<<"Total_Above: "<<matmultiplicationTime+cubeTime+transposeMulTime+columnviseTime+subtractTime+copyTime+symDecorelationTime+limTime+saveWTime<<endl;
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
	ica.fit_transform(X,_S,W);
	
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
	
