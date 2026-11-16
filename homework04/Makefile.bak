CC =  gcc
CPP = g++
NVCC =  /packages/cuda/11.2.0/bin/nvcc
CCFLAGS = -O3 -fopenmp -lm -std=c++11
CFLAGS += -O3 -fopenmp -lm -std=c11

CUDA_INSTALL_PATH ?= /packages/cuda/10.2
TOOLKIT_INSTALL_PATH ?= /packages/cuda/11.2.0/samples
CUDA_LDFLAGS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcurand -lcusolver
NVCCFLAGS = -gencode arch=compute_80,code=sm_80 #--ptxas-options=-v
NVCCINC += -I$(CUDA_INSTALL_PATH)/include 
NVCCINC += -I$(TOOLKIT_INSTALL_PATH)/common/inc

SRC  = $(notdir $(wildcard *.cc)) $(notdir $(wildcard *.cu)) $(notdir $(wildcard *.c))

OBJS = $(addsuffix .o, $(basename $(SRC)))
EXEC = spmv

all: ${EXEC}

${EXEC}: ${OBJS}
	${CPP} ${CFLAGS} ${LDFLAGS} $^ -o $@ ${LIBS} ${CUDA_LDFLAGS}

%.o : %.c
	$(CC) ${CFLAGS} ${INC} -c $< -o $@ ${LIBS} ${CUDA_LD_FLAGS} ${NVCCINC}

%.o : %.cc
	$(CPP) ${CCFLAGS} ${INC} -c $< -o $@ ${LIBS} ${CUDA_LD_FLAGS} ${NVCCINC}

%.o : %.cu
	$(NVCC) ${NVCCCFLAGS} ${INC} -c $< -o $@ ${CUDA_LD_FLAGS} ${NVCCFLAGS} ${NVCCINC}

clean:
	rm -f *.o *.d *~ *.a *.so *.s ${EXEC}

