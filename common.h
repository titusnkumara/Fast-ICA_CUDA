struct CUDA_variables{
	float * X1;
	float * X1Transpose;
	float * W;
	float * W1;
	float * w_init;
	float * product;
	float * gwtx;
	float * g_wtx;
	float * cubeD;
	float * g_wtx_X1_transpose;
	float * gwtx_into_W;
	float * w_init_w_init_T;
	float * eigenValues;
	float * eigenVectors;
	float * W1intoWT;
	
	int* it_num;
	int* rot_num;
	float *bw;
	float *zw;
	
	float * eigenRowWise;
	float * diagonal;
	float * answer;
	
	float * tmp_w_init;
	
	float * d_singleArray;
	float * d_computeArray;
	
	float * hostpointer;
	
	};

struct preprocessVariableList{
	double * d_X;
	float * d_Xf;
	double * d_X_tr;
	float * d_X_trf;
	float * d_VT;
	float * d_VTT;
	float * d_S;
	float * d_w_init;
	float * d_w_init_T;
	
	
	
};

typedef struct CUDA_variables cudaVar;
typedef struct preprocessVariableList preprocessVariables;

