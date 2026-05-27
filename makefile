CONFIG ?=
ifneq (${CONFIG},)
include ${CONFIG}
endif

WRAPPER_GOALS = dmplex dmplex-clean direct-cuda direct-cuda-clean cuda cuda-clean kokkos-cuda kokkos-cuda-clean kokkos-openmp kokkos-openmp-clean
NO_PETSC_GOALS = cleanobj ${WRAPPER_GOALS}
SKIP_PETSC_CONF = $(if ${MAKECMDGOALS},$(if $(filter-out ${NO_PETSC_GOALS},${MAKECMDGOALS}),,1))

ifeq (${SKIP_PETSC_CONF},)
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
endif

SRCDIR      = src
INCDIR      = include
BUILDDIR    = build
OBJDIR      = ${BUILDDIR}/obj
USE_CUDA   ?= 1
USE_KOKKOS ?= 0
ifeq (${USE_CUDA},1)
BACKEND ?= cuda
else ifeq (${USE_KOKKOS},1)
BACKEND ?= kokkos
else
BACKEND ?= cpu
endif
TEST_EXE    = ${BUILDDIR}/testt-${BACKEND}

ifeq (${USE_CUDA},1)
ifeq (${USE_KOKKOS},1)
$(error USE_CUDA=1 and USE_KOKKOS=1 are mutually exclusive; choose direct CUDA or Kokkos)
endif
endif

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
KOKKOS_CXX ?= $(CXXLINKER)
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
	${SRCDIR}/dmplex_geom.c \
	${SRCDIR}/manufactured_active_strain.c \
	${SRCDIR}/cuda_processes.c \
	${SRCDIR}/kokkos_processes.c \
	${SRCDIR}/cuda_stub.c \
	${SRCDIR}/kokkos_stub.c

SOURCECPP =
SOURCECU  =

ifeq (${USE_CUDA},1)
SOURCEC := $(filter-out ${SRCDIR}/cuda_stub.c,${SOURCEC})
SOURCECPP += ${SRCDIR}/cuda_wrapper.cpp
SOURCECU  += ${SRCDIR}/cuda_kernel.cu
CFLAGS    += -DUSE_CUDA -DBIOFEM_BACKEND_CUDA
CPPFLAGS  += -DUSE_CUDA -DBIOFEM_BACKEND_CUDA
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

ifeq (${USE_KOKKOS},1)
SOURCEC := $(filter-out ${SRCDIR}/kokkos_stub.c,${SOURCEC})
SOURCECPP += ${SRCDIR}/kokkos_geom.cpp
CFLAGS    += -DUSE_KOKKOS -DBIOFEM_BACKEND_KOKKOS
CPPFLAGS  += -DUSE_KOKKOS -DBIOFEM_BACKEND_KOKKOS
KOKKOS_CXXFLAGS ?=
KOKKOS_LDFLAGS ?=
KOKKOS_LIBS ?= -lkokkoscore -lkokkoscontainers -lkokkosalgorithms
CXXFLAGS  += ${KOKKOS_CXXFLAGS}
LIBS      += ${KOKKOS_LIBS}
LDFLAGS   += ${KOKKOS_LDFLAGS}
endif

OBJSC = ${patsubst ${SRCDIR}/%.c,${OBJDIR}/%.o,${SOURCEC}}
OBJSCXX = ${patsubst ${SRCDIR}/%.cpp,${OBJDIR}/%.o,${SOURCECPP}}
OBJSCU  = ${patsubst ${SRCDIR}/%.cu,${OBJDIR}/%.o,${SOURCECU}}

LIBBASE = libpetscmat

${OBJDIR}/%.o: ${SRCDIR}/%.c | ${OBJDIR}
	$(CLINKER) -c ${CFLAGS} ${CPPFLAGS} $< -o $@

${OBJDIR}/%.o: ${SRCDIR}/%.cpp | ${OBJDIR}
	$(KOKKOS_CXX) -c ${CFLAGS} ${CXXFLAGS} ${CPPFLAGS} $< -o $@

${OBJDIR}/%.o: ${SRCDIR}/%.cu | ${OBJDIR}
	$(NVCC) -c ${CUDAFLAGS} ${CPPFLAGS} $< -o $@

${OBJDIR}:
	mkdir -p ${OBJDIR}

${TEST_EXE}: ${OBJSC} ${OBJSCXX} ${OBJSCU}
	-$(KOKKOS_CXX) -o ${TEST_EXE} ${CFLAGS} ${CXXFLAGS} ${OBJSC} ${OBJSCXX} ${OBJSCU} ${PETSC_LIB} ${LDFLAGS} ${LIBS}

testt: ${TEST_EXE}

asan: CFLAGS += ${SAN_FLAGS}
asan: LIBS += -fsanitize=address,undefined
asan: cleanobj testt

.PHONY: testt asan cleanobj dmplex dmplex-clean direct-cuda direct-cuda-clean cuda cuda-clean kokkos-cuda kokkos-cuda-clean kokkos-openmp kokkos-openmp-clean

cleanobj:
	rm -rf ${BUILDDIR} *.o testt testt-*

dmplex:
	$(MAKE) CONFIG=config/dmplex.mk testt

dmplex-clean:
	$(MAKE) CONFIG=config/dmplex.mk cleanobj testt

direct-cuda:
	$(MAKE) CONFIG=config/direct-cuda.mk testt

direct-cuda-clean:
	$(MAKE) CONFIG=config/direct-cuda.mk cleanobj testt

cuda: direct-cuda

cuda-clean: direct-cuda-clean

kokkos-cuda:
	$(MAKE) CONFIG=config/kokkos-cuda.mk testt

kokkos-cuda-clean:
	$(MAKE) CONFIG=config/kokkos-cuda.mk cleanobj testt

kokkos-openmp:
	$(MAKE) CONFIG=config/kokkos-openmp.mk testt

kokkos-openmp-clean:
	$(MAKE) CONFIG=config/kokkos-openmp.mk cleanobj testt

ifeq (${SKIP_PETSC_CONF},)
include ${PETSC_DIR}/lib/petsc/conf/test
endif
