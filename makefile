CONFIG ?=
ifneq (${CONFIG},)
include ${CONFIG}
endif

WRAPPER_GOALS = dmplex dmplex-clean direct-cuda direct-cuda-clean cuda cuda-clean kokkos-cuda kokkos-cuda-clean kokkos-openmp kokkos-openmp-clean
NO_PETSC_GOALS = cleanobj clean build build-cuda build-kokkos-cuda build-kokkos-openmp install ${WRAPPER_GOALS}
SKIP_PETSC_CONF = $(if ${MAKECMDGOALS},$(if $(filter-out ${NO_PETSC_GOALS},${MAKECMDGOALS}),,1))

ifeq (${SKIP_PETSC_CONF},)
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
endif

SRCDIR      = src
INCDIR      = include
BUILDDIR    = build
USE_CUDA   ?= 1
USE_KOKKOS ?= 0
ifeq (${USE_CUDA},1)
BACKEND ?= cuda
else ifeq (${USE_KOKKOS},1)
BACKEND ?= kokkos
else
BACKEND ?= cpu
endif
# Each backend gets its own object directory (obj-<backend>, not a single
# shared obj/) so building one backend never leaves another backend's
# executable linked against stale objects compiled with the wrong
# USE_CUDA/USE_KOKKOS flags -- and so testt-dmplex, testt-cuda,
# testt-kokkos-cuda, testt-kokkos-openmp can all exist and be submitted as
# jobs at the same time, since building one no longer requires deleting
# the others. OBJDIR uses plain '=' (lazy/recursive expansion), so this is
# safe even though BACKEND is only fully resolved above.
OBJDIR      = ${BUILDDIR}/obj-${BACKEND}
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
	${SRCDIR}/lv_geometry.c \
	${SRCDIR}/lv_geometry_unstructured.c \
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

.PHONY: testt asan cleanobj cleanall clean dmplex dmplex-clean direct-cuda direct-cuda-clean cuda cuda-clean kokkos-cuda kokkos-cuda-clean kokkos-openmp kokkos-openmp-clean build-all build build-cuda build-kokkos-cuda build-kokkos-openmp install

# Removes only the current backend's own object dir + executable (safe to
# run before rebuilding one backend without disturbing any other backend's
# already-built binary). For a full nuke of every backend's build output,
# use `cleanall` instead.
cleanobj:
	rm -rf ${OBJDIR} ${TEST_EXE}

cleanall:
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

# Builds every legacy backend in one go. Safe to run any time -- each
# backend's objects and executable live in their own obj-<backend>/ and
# testt-<backend>, so this never deletes a binary another job is using.
build-all: dmplex direct-cuda kokkos-cuda kokkos-openmp

# ── cmake-based build and install ────────────────────────────────────────
clean::
	rm -rf build

build:
	cmake -B build -DBIOFEM_DMPLEX=ON -DBIOFEM_CUDA=OFF -DBIOFEM_KOKKOS_CUDA=OFF -DBIOFEM_KOKKOS_OPENMP=OFF
	cmake --build build

build-cuda:
	cmake -B build -DBIOFEM_DMPLEX=OFF -DBIOFEM_CUDA=ON -DBIOFEM_KOKKOS_CUDA=OFF -DBIOFEM_KOKKOS_OPENMP=OFF
	cmake --build build

build-kokkos-cuda:
	cmake -B build -DBIOFEM_DMPLEX=OFF -DBIOFEM_CUDA=OFF -DBIOFEM_KOKKOS_CUDA=ON -DBIOFEM_KOKKOS_OPENMP=OFF
	cmake --build build

build-kokkos-openmp:
	cmake -B build -DBIOFEM_DMPLEX=OFF -DBIOFEM_CUDA=OFF -DBIOFEM_KOKKOS_CUDA=OFF -DBIOFEM_KOKKOS_OPENMP=ON
	cmake --build build

install:
	cmake --install build
	@if ! echo "$$PATH" | tr ':' '\n' | grep -qx "$$HOME/.local/bin"; then \
		echo ""; \
		echo "  One-time setup: add ~/.local/bin to your PATH."; \
		echo "  Run this once:"; \
		echo ""; \
		echo "    echo 'export PATH=\"\$$HOME/.local/bin:\$$PATH\"' >> ~/.bashrc && source ~/.bashrc"; \
		echo ""; \
	else \
		echo "  biofem is ready. Try: biofem --help"; \
	fi

ifeq (${SKIP_PETSC_CONF},)
include ${PETSC_DIR}/lib/petsc/conf/test
endif
