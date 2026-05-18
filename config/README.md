# Build Configs

Use `CONFIG=...` to select a repeatable build mode:

```bash
make cleanobj
make CONFIG=config/direct-cuda.mk
make CONFIG=config/cpu.mk
make CONFIG=config/kokkos-openmp.mk
make CONFIG=config/kokkos-cuda.mk KOKKOS_CXX=nvcc_wrapper CUDA_ARCH=sm_80
```

These are Make fragments, not JSON, because the project is built by Make and
Make can include `.mk` files directly.

Runtime flags are separate from build flags. For example:

```bash
# direct CUDA preprocessing path
./build/testt-cuda -cuda_process 1

# Kokkos subdivision geometry path
./build/testt-kokkos-cuda -muscle_activation 1 -kokkos_process 1
```

Kokkos backend selection is mostly decided when Kokkos itself is built. This
project selects whether to compile the Kokkos source, while the Kokkos compiler
wrapper/library determines whether execution maps to OpenMP, CUDA, etc.
