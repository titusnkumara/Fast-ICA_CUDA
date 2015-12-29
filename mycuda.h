
cudaVar initializeCuda(preprocessVariables* DevicePointers, cudaVar cudaVariables,int n,int p);
void matrixMultiplyonGPU(float * d_A, float * d_B, float * d_C,int n,int p);
void copyBackProductfromCUDA(MatrixXf& product,float * from);
void cubeOnGPU(cudaVar cudaVariables,int n,int p);
void copyBackCubefromCUDA(MatrixXf& cube,float * from);
void copyBackCubeDerivationfromCUDA(MatrixXf& cubeDerivation,float * from);
void copyBackX1fromCUDA(MatrixXf& X1,float * from);

void copyBackTransposeMulfromCUDA(MatrixXf& tr,float * from);
void multiplyColumnViseOnGPU(cudaVar cudaVariables,int n,int p);
void subtractOnGPU(cudaVar cudaVariables,int n);
void saveW1inGPU(MatrixXd& W1,cudaVar cudaVariables,int n);
void copyBackWfromCUDA(MatrixXf& W,float * from);
void matrixMultiplyTransposeImprovedonGPU(cudaVar cudaVariables,float p_,int n,int p);
void create_blas_handler();
void destroy_blas_handler();
cudaVar findEigenOnCuda(cudaVar cudaVariables);
int runSVDonCUDA(float * A,preprocessVariables* DevicePointers,int ROWS,int COLS);

void getMeanNormalizeOnCUDA(MatrixXd& X,int n,int p,preprocessVariables* DevicePointers );
void devideVTbySingularValues(float * d_VT,float * d_VTT, float * d_S,int n);
void multiplyOnGPU_K_X(preprocessVariables* DevicePointers,int n,int p);


void sym_decorrelation_cuda(preprocessVariables* DevicePointers,int n);
void memSetForSymDecorrelationCUDA(MatrixXf& w_init,preprocessVariables* DevicePointers,int n);
void copyKtoHost(MatrixXd& K,preprocessVariables* DevicePointers,int n);

void copyBackW1fromCUDA(MatrixXd& W1,float * d_W1,int n);
