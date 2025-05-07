include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ALL: testt 

ifdef TEC360HOME
CFLAGS       = -I${TEC360HOME}/include/ -DTECIO=1
LIBS         = ${TEC360HOME}/lib/tecio64.a -lstdc++
else
CFLAGS       = -g
LIBS         =
endif

# PYTHON_INC   = -I/scratch/user/hgeshani/.conda/envs/pyt/include/python3.11
# PYTHON_LIB   = -L/scratch/user/hgeshani/.conda/envs/pyt/lib -lpython3.11 -lcrypt -ldl -lm -lpthread -lutil 

PYTHON_INC   = -I/scratch/user/hgeshani/.conda/envs/pyt38/include/python3.8
PYTHON_LIB   = -L/scratch/user/hgeshani/.conda/envs/pyt38/lib -lpython3.8 -lcrypt -ldl -lm -lpthread -lutil 
OPENMP_FLAGS = -fopenmp

CFLAGS      += ${PYTHON_INC} ${OPENMP_FLAGS}
LIBS        += ${PYTHON_LIB} ${OPENMP_FLAGS}

CFLAGS      += ${PYTHON_INC}
LIBS        += ${PYTHON_LIB}

CLINKER     = mpicc
LDFLAGS      = 
FFLAGS       =
CPPFLAGS     =  
FPPFLAGS     =
LOCDIR       = 
MANSEC       = SNES

LIBFLAG      =

SOURCEC = main.c io.c membrane.c math.c external.c bending.c material.c bcs.c contact.c PyCaller.c inverse.c

OBJSC = main.o io.o membrane.o math.o external.o bending.o material.o bcs.o contact.o PyCaller.o inverse.o

LIBBASE = libpetscmat




testt: ${OBJSC}
	-$(CLINKER) -o testt ${CFLAGS} ${OBJSC} ${PETSC_LIB} ${LIBS}


cleanobj:
	rm -f *.o

include ${PETSC_DIR}/lib/petsc/conf/test
