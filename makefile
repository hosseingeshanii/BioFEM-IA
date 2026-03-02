include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

SRCDIR      = src
INCDIR      = include
BUILDDIR    = build
OBJDIR      = ${BUILDDIR}/obj
TEST_EXE    = ${BUILDDIR}/testt

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
	${SRCDIR}/active_strain.c

OBJSC = ${patsubst ${SRCDIR}/%.c,${OBJDIR}/%.o,${SOURCEC}}

LIBBASE = libpetscmat

${OBJDIR}/%.o: ${SRCDIR}/%.c | ${OBJDIR}
	$(CLINKER) -c ${CFLAGS} ${CPPFLAGS} $< -o $@

${OBJDIR}:
	mkdir -p ${OBJDIR}

${TEST_EXE}: ${OBJSC}
	-$(CLINKER) -o ${TEST_EXE} ${CFLAGS} ${OBJSC} ${PETSC_LIB} ${LIBS}

testt: ${TEST_EXE}

asan: CFLAGS += ${SAN_FLAGS}
asan: LIBS += -fsanitize=address,undefined
asan: cleanobj testt


cleanobj:
	rm -rf ${BUILDDIR} *.o testt

include ${PETSC_DIR}/lib/petsc/conf/test
