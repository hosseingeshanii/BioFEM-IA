cmake_minimum_required(VERSION 3.18)
project(BioFEM-IA VERSION 0.1.0 LANGUAGES C CXX)

# ── Default install prefix: ~/.local (standard user-local location) ──────
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local"
      CACHE PATH "Install prefix" FORCE)
endif()
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

# ── Build options (pass with -D on the cmake command line) ───────────────
option(BIOFEM_DMPLEX        "Build DMPlex/MPI backend"    ON)
option(BIOFEM_CUDA          "Build direct CUDA backend"   OFF)
option(BIOFEM_KOKKOS_CUDA   "Build Kokkos CUDA backend"   OFF)
option(BIOFEM_KOKKOS_OPENMP "Build Kokkos OpenMP backend" OFF)

# ── Find MPI ─────────────────────────────────────────────────────────────
find_package(MPI REQUIRED)

# ── Find PETSc ───────────────────────────────────────────────────────────
if(NOT DEFINED ENV{PETSC_DIR})
  message(FATAL_ERROR
    "PETSC_DIR is not set.\n"
    "Load your PETSc module first:  source load-env.sh")
endif()
set(PETSC_DIR "$ENV{PETSC_DIR}" CACHE PATH "PETSc installation directory")

# PETSc >= 3.14 ships a CMake config; fall back to pkg-config if not found.
find_package(PETSc QUIET
  HINTS
    ${PETSC_DIR}/lib/cmake/PETSc
    ${PETSC_DIR}/$ENV{PETSC_ARCH}/lib/cmake/PETSc
)
if(PETSc_FOUND)
  message(STATUS "PETSc ${PETSc_VERSION} found via CMake config")
else()
  find_package(PkgConfig REQUIRED)
  if(DEFINED ENV{PETSC_ARCH})
    set(ENV{PKG_CONFIG_PATH}
        "${PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  else()
    set(ENV{PKG_CONFIG_PATH}
        "${PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  endif()
  pkg_search_module(PETSc REQUIRED petsc)
  # Wrap into an imported target so every backend uses the same syntax.
  add_library(PETSc::PETSc INTERFACE IMPORTED)
  target_include_directories(PETSc::PETSc INTERFACE ${PETSc_INCLUDE_DIRS})
  target_link_libraries(PETSc::PETSc INTERFACE ${PETSc_LDFLAGS})
  message(STATUS "PETSc found via pkg-config")
endif()

# ── Sources shared by all backends ───────────────────────────────────────
# (matches the SOURCEC list in the Makefile, minus backend-specific stubs)
set(BIOFEM_COMMON
  src/main.c
  src/io.c
  src/membrane.c
  src/math.c
  src/external.c
  src/bending.c
  src/material.c
  src/bcs.c
  src/contact.c
  src/inverse.c
  src/active_strain.c
  src/dmplex_geom.c
  src/lv_geometry.c
  src/manufactured_active_strain.c
  src/cuda_processes.c
  src/kokkos_processes.c
)

# Helper: apply flags every backend needs
function(biofem_common_setup target)
  target_include_directories(${target} PRIVATE include)
  target_compile_options(${target} PRIVATE -fopenmp -g)
  target_link_libraries(${target} PRIVATE MPI::MPI_C PETSc::PETSc -fopenmp m)
endfunction()

# ── DMPlex / MPI backend ─────────────────────────────────────────────────
if(BIOFEM_DMPLEX)
  add_executable(biofem-dmplex
    ${BIOFEM_COMMON}
    src/cuda_stub.c
    src/kokkos_stub.c
  )
  biofem_common_setup(biofem-dmplex)
  install(TARGETS biofem-dmplex DESTINATION bin)
  message(STATUS "Backend enabled: dmplex")
endif()

# ── Direct CUDA backend ──────────────────────────────────────────────────
if(BIOFEM_CUDA)
  enable_language(CUDA)
  add_executable(biofem-cuda
    ${BIOFEM_COMMON}
    src/cuda_wrapper.cpp
    src/cuda_kernel.cu
    src/kokkos_stub.c
  )
  biofem_common_setup(biofem-cuda)
  target_compile_definitions(biofem-cuda PRIVATE USE_CUDA BIOFEM_BACKEND_CUDA)
  target_link_libraries(biofem-cuda PRIVATE -lstdc++ -lcudart)
  if(DEFINED ENV{CUDA_HOME})
    target_include_directories(biofem-cuda PRIVATE $ENV{CUDA_HOME}/include)
    target_link_directories(biofem-cuda PRIVATE $ENV{CUDA_HOME}/lib64)
  endif()
  if(DEFINED ENV{CUDA_ARCH})
    set_property(TARGET biofem-cuda PROPERTY CUDA_ARCHITECTURES $ENV{CUDA_ARCH})
  endif()
  install(TARGETS biofem-cuda DESTINATION bin)
  message(STATUS "Backend enabled: cuda")
endif()

# ── Kokkos CUDA backend ──────────────────────────────────────────────────
if(BIOFEM_KOKKOS_CUDA)
  find_package(Kokkos REQUIRED
    HINTS $ENV{SCRATCH}/kokkos-install $ENV{KOKKOS_INSTALL})
  add_executable(biofem-kokkos-cuda
    ${BIOFEM_COMMON}
    src/kokkos_geom.cpp
    src/cuda_stub.c
  )
  biofem_common_setup(biofem-kokkos-cuda)
  target_compile_definitions(biofem-kokkos-cuda PRIVATE USE_KOKKOS BIOFEM_BACKEND_KOKKOS)
  target_link_libraries(biofem-kokkos-cuda PRIVATE Kokkos::kokkos)
  install(TARGETS biofem-kokkos-cuda DESTINATION bin)
  message(STATUS "Backend enabled: kokkos-cuda")
endif()

# ── Kokkos OpenMP backend ────────────────────────────────────────────────
if(BIOFEM_KOKKOS_OPENMP)
  find_package(Kokkos REQUIRED
    HINTS $ENV{SCRATCH}/kokkos-install $ENV{KOKKOS_INSTALL})
  add_executable(biofem-kokkos-openmp
    ${BIOFEM_COMMON}
    src/kokkos_geom.cpp
    src/cuda_stub.c
  )
  biofem_common_setup(biofem-kokkos-openmp)
  target_compile_definitions(biofem-kokkos-openmp PRIVATE USE_KOKKOS BIOFEM_BACKEND_KOKKOS)
  target_link_libraries(biofem-kokkos-openmp PRIVATE Kokkos::kokkos)
  install(TARGETS biofem-kokkos-openmp DESTINATION bin)
  message(STATUS "Backend enabled: kokkos-openmp")
endif()

# ── Install CLI script as a symlink ─────────────────────────────────────
# A symlink means:
#   - edits to bin/biofem are live immediately (no reinstall needed)
#   - after cmake --build, biofem finds the new binary in build/ automatically
#   - compiled executables do NOT need to be copied to the install prefix
install(CODE "
  set(link  \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/biofem\")
  set(target \"${CMAKE_SOURCE_DIR}/bin/biofem\")
  message(STATUS \"Installing symlink: \${link} -> \${target}\")
  file(CREATE_LINK \"\${target}\" \"\${link}\" SYMBOLIC)
")
