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
	float *d_A;
	float *h_A;
	
	float *h_U;
	float *h_V;
	float *h_S;
	
	float *d_U;
	float *d_V;
	float *d_S;
};

typedef struct CUDA_variables cudaVar;
typedef struct preprocessVariableList preprocessVariables;

