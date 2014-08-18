CC=gcc
CCPLUS=g++
NVCC=nvcc
CFLAGS=-DNDEBUG -O3 

INCLUDES=-I${CULA_INC_PATH} -I${CUDA_INC_PATH}
LIBPATH64=-L${CULA_LIB_PATH_64} -L${CUDA_LIB_PATH_64}

LIBS= -lcula_lapack_basic -lcublas -lcudart -liomp5 -lm

all: run 

run: cudaSolver
	./cudaSolver

cudaSolver: expcuda.o cudaSolver.o initA.o
	$(NVCC) -m64 -o cudaSolver cudaSolver.o expcuda.o initA.o $(LIBPATH64) $(LIBS)

initA.o: initA.cu
	$(NVCC) -c -m64 -o initA.o initA.cu $(CFLAGS) $(INCLUDES)

expcuda.o: expcuda.cu
	$(NVCC) -c -m64 -o expcuda.o expcuda.cu $(CFLAGS) $(INCLUDES)

cudaSolver.o: cudaSolver.cu
	$(NVCC) -c -m64 -o cudaSolver.o cudaSolver.cu $(CFLAGS) $(INCLUDES)

clean:
	rm -rf *o cudaSolver
