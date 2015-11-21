make clean
rm ../output/output.txt
make

nvcc -o out cudapart.cu main.o -I "../" -lcublas

strip out

rm main.o
