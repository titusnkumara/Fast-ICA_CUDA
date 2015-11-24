
cudaVar initializeCuda(MatrixXd& W,MatrixXd& X1,MatrixXd& w_init, cudaVar cudaVariables,int n,int p);
//preprocessVariables initializeCudaForPreprocess(MatrixXd& S,preprocessVariables preprocessData,int n,int p);
void matrixMultiplyonGPU(float * d_A, float * d_B, float * d_C,int n,int p);
void copyBackProductfromCUDA(MatrixXf& product,float * from);
void cubeOnGPU(cudaVar cudaVariables,int n,int p);
void copyBackCubefromCUDA(MatrixXf& cube,float * from);
void copyBackCubeDerivationfromCUDA(MatrixXf& cubeDerivation,float * from);
void copyBackX1fromCUDA(MatrixXf& X1,float * from);

void copyBackTransposeMulfromCUDA(MatrixXf& tr,float * from);
void multiplyColumnViseOnGPU(cudaVar cudaVariables,int n,int p);
void subtractOnGPU(cudaVar cudaVariables,int n);
void copyBackW_initfromCUDA(MatrixXd& w_init,MatrixXf& tmp,float * from,float * hostpointer,float * tmp_w_init,int n);
void saveW1inGPU(MatrixXd& W1,cudaVar cudaVariables,int n);
void copyBackWfromCUDA(MatrixXf& W,float * from);
void matrixMultiplyTransposeImprovedonGPU(cudaVar cudaVariables,float p_,int n,int p);
void create_blas_handler();
void destroy_blas_handler();
cudaVar findEigenOnCuda(cudaVar cudaVariables);
int runSVDonCUDA(MatrixXd& input,VectorXd& singularValue,MatrixXd& singularVectors,int ROWS,int COLS);

