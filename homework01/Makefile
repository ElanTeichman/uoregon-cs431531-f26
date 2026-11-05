CC =  /usr/bin/gcc
CFLAGS  = -Wall -g -std=c99
CFLAGS += -O3 -fopenmp -lm
CFLAGS += -D_POSIX_C_SOURCE=200112L
LIBS =

SRC  = $(notdir $(wildcard *.c)) 
OBJS = $(addsuffix .o, $(basename $(SRC)))
EXEC = pi

all: ${EXEC}

${EXEC}: ${OBJS}
	${CC} ${CFLAGS} ${LDFLAGS} $^ -o $@ ${LIBS} ${CUDA_LDFLAGS}

%.o : %.c
	$(CC) ${CFLAGS} ${INC} ${NVCCINC} -c $< -o $@ ${LIBS}

clean:
	rm -f *.o *.d *~ *.a *.so *.s ${EXEC}

