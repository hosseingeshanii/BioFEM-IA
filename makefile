include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ALL: testt 

ifdef TEC360HOME
CFLAGS       = -I${TEC360HOME}/include/ -DTECIO=1
LIBS         = ${TEC360HOME}/lib/tecio64.a -lstdc++
else
CFLAGS       += -g
# CFLAGS 		 += -DCHECK_JACOBIAN	
LIBS         =
endif


OPENMP_FLAGS = -fopenmp
SAN_FLAGS    = -O0 -g3 -fno-omit-frame-pointer -fsanitize=address,undefined

CFLAGS      +=  ${OPENMP_FLAGS}
LIBS        +=  ${OPENMP_FLAGS}


CLINKER     = mpicc
LDFLAGS      = 
FFLAGS       =
CPPFLAGS     =  
FPPFLAGS     =
LOCDIR       = 
MANSEC       = SNES

LIBFLAG      =

SOURCEC = main.c io.c membrane.c math.c external.c bending.c material.c bcs.c contact.c inverse.c active_strain.c

OBJSC = main.o io.o membrane.o math.o external.o bending.o material.o bcs.o contact.o inverse.o active_strain.o

LIBBASE = libpetscmat




testt: ${OBJSC}
	-$(CLINKER) -o testt ${CFLAGS} ${OBJSC} ${PETSC_LIB} ${LIBS}

asan: CFLAGS += ${SAN_FLAGS}
asan: LIBS += -fsanitize=address,undefined
asan: cleanobj testt


cleanobj:
	rm -f *.o testt

include ${PETSC_DIR}/lib/petsc/conf/test
