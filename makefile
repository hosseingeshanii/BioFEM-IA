include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

SRCDIR      = src
INCDIR      = include
BUILDDIR    = build
OBJDIR      = ${BUILDDIR}/obj
TEST_EXE    = ${BUILDDIR}/testt
USE_CUDA   = 1

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
CFLAGS      += -I${INCDIR}


CLINKER     = mpicc
CXXLINKER   = mpicxx
NVCC        ?= nvcc
LDFLAGS      = 
FFLAGS       =
CPPFLAGS     =  
FPPFLAGS     =
LOCDIR       = 
MANSEC       = SNES

LIBFLAG      =

SOURCEC = \
	${SRCDIR}/main.c \
	${SRCDIR}/io.c \
	${SRCDIR}/membrane.c \
	${SRCDIR}/math.c \
	${SRCDIR}/external.c \
	${SRCDIR}/bending.c \
	${SRCDIR}/material.c \
	${SRCDIR}/bcs.c \
	${SRCDIR}/contact.c \
	${SRCDIR}/inverse.c \
	${SRCDIR}/active_strain.c \
	${SRCDIR}/manufactured_active_strain.c \
	${SRCDIR}/cuda_stub.c

SOURCECPP =
SOURCECU  =

ifeq (${USE_CUDA},1)
SOURCEC := $(filter-out ${SRCDIR}/cuda_stub.c,${SOURCEC})
SOURCECPP += ${SRCDIR}/cuda_wrapper.cpp
SOURCECU  += ${SRCDIR}/cuda_kernel.cu
CFLAGS    += -DUSE_CUDA
CPPFLAGS  += -DUSE_CUDA
LIBS      += -lstdc++
CUDA_ARCH ?=
CUDAFLAGS += -I${INCDIR} -Xcompiler -fPIC
ifneq (${CUDA_HOME},)
CUDAFLAGS += -I${CUDA_HOME}/include
LDFLAGS   += -L${CUDA_HOME}/lib64
endif
ifneq (${CUDA_ARCH},)
CUDAFLAGS += -arch=${CUDA_ARCH}
endif
LIBS      += -lcudart
endif

OBJSC = ${patsubst ${SRCDIR}/%.c,${OBJDIR}/%.o,${SOURCEC}}
OBJSCXX = ${patsubst ${SRCDIR}/%.cpp,${OBJDIR}/%.o,${SOURCECPP}}
OBJSCU  = ${patsubst ${SRCDIR}/%.cu,${OBJDIR}/%.o,${SOURCECU}}

LIBBASE = libpetscmat

${OBJDIR}/%.o: ${SRCDIR}/%.c | ${OBJDIR}
	$(CLINKER) -c ${CFLAGS} ${CPPFLAGS} $< -o $@

${OBJDIR}/%.o: ${SRCDIR}/%.cpp | ${OBJDIR}
	$(CXXLINKER) -c ${CFLAGS} ${CPPFLAGS} $< -o $@

${OBJDIR}/%.o: ${SRCDIR}/%.cu | ${OBJDIR}
	$(NVCC) -c ${CUDAFLAGS} ${CPPFLAGS} $< -o $@

${OBJDIR}:
	mkdir -p ${OBJDIR}

${TEST_EXE}: ${OBJSC} ${OBJSCXX} ${OBJSCU}
	-$(CXXLINKER) -o ${TEST_EXE} ${CFLAGS} ${OBJSC} ${OBJSCXX} ${OBJSCU} ${PETSC_LIB} ${LDFLAGS} ${LIBS}

testt: ${TEST_EXE}

asan: CFLAGS += ${SAN_FLAGS}
asan: LIBS += -fsanitize=address,undefined
asan: cleanobj testt


cleanobj:
	rm -rf ${BUILDDIR} *.o testt

include ${PETSC_DIR}/lib/petsc/conf/test
