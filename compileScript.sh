make clean
rm ../output/output.txt
make

nvcc -m64 -o out cudapart.cu ./lib/helpers.cu main.o -I "../" -lcublas -lcusolver -lcula_lapack -lpthread -liomp5 -I${CULA_INC_PATH} -L${CULA_LIB_PATH_32} -L${CULA_LIB_PATH_64}


#nvcc -m64 -o gesvd mySVD.cpp -lcula_lapack -lpthread -liomp5 -I${CULA_INC_PATH} -L${CULA_LIB_PATH_32} -L${CULA_LIB_PATH_64}

strip out

rm main.o
