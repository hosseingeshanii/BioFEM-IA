# BioFEM-IA

Finite element solver for biological membranes with active strain. Supports MPI (DMPlex), direct CUDA, and Kokkos (CUDA/OpenMP) backends.

## Requirements

- PETSc >= 3.14 with MPI
- GCC >= 9, CMake >= 3.18
- CUDA toolkit (optional, for CUDA/Kokkos-CUDA backends)
- Kokkos (optional, for Kokkos backends)
- Python >= 3.8 with numpy and scipy (for `biofem sweep`)

## Build and install

```bash
# 1. Load environment (PETSc, MPI, CUDA modules)
source load-env.sh

# 2. Compile — choose your backend
make build              # DMPlex/MPI (default)
make build-cuda         # direct CUDA
make build-kokkos-cuda  # Kokkos CUDA
make build-kokkos-openmp

# 3. Install the biofem CLI (one time only)
make install

# 4. Add to PATH if prompted (one time only)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

After a code change, just `make build` — no reinstall needed.

## Site configuration

Copy and edit the site config to set your SLURM account and cluster modules:

```bash
cp config/site.cfg.template config/site.cfg
```

This is optional — you can also pass `--account` on the command line.

## Usage

### Run a single case

```bash
cd /path/to/case/dir    # must contain control.dat and input/
biofem run --backend dmplex --np 8
biofem run --backend cuda --dry-run   # preview jobfile without submitting
```

On a SLURM cluster the job is automatically submitted via `sbatch`. If you are already in an interactive allocation, it runs directly with `mpirun`.

### Parameter sweeps

```bash
# Core count sweep — fixed mesh, varying MPI tasks
biofem sweep --backend dmplex \
    --nx 64 --ny 48 \
    --ntasks 1,2,4,8,16,32 \
    --output-dir ./core_sweep

# Mesh size sweep — fixed core count, varying mesh
biofem sweep --backend dmplex \
    --nx 32,64,128,256 --ny 24,48,96,192 \
    --ntasks 8 \
    --output-dir ./mesh_sweep

# Preview without submitting
biofem sweep ... --dry-run
```

Generates one case directory per mesh size under `--output-dir`, with a separate `jobfile_np{N}.slurm` per core count. Skips combinations where elements-per-task falls below the minimum (default 100).

### Post-processing

```bash
cd /path/to/case/dir
biofem post convergence                 # ||R||, reduction and Newton its per step, from the solver log (*.out)
biofem post lv                          # LV: volume/EF, |u|, wall thickening, shortening, torsion vs time
biofem post lv --ref ref.json           # add reference values (label, EF, SV, thickening, shortening, torsion)
biofem post field --dir . --field disp --out-dir output   # average of one VTK field vs time
```

Each command writes a PNG and a CSV next to it (`--out`), and takes `--dir` several times to overlay cases.
`post lv` reads `surface00_*.vtk`, `lv_volume00.dat` and, with `-write_3d_shell 1`, `shell3d00_*.vtk`;
lengths are shown in cm and volumes in ml (`--solver-unit` gives the solver's unit, default m).
`biofem postprocess` is the older name of `post field`.

### All options

```bash
biofem --help
biofem run --help
biofem sweep --help
biofem post --help
```

## Project layout

```
BioFEM-IA/
├── src/            solver source (C, CUDA, C++)
├── include/        headers
├── config/         per-backend make configs and site.cfg template
├── examples/
│   └── dmplex_test/    canonical single-case example
├── bin/
│   └── biofem      CLI launcher (Python)
├── python/biofem/  CLI code: run, sweep, post (field, convergence, lv)
├── CMakeLists.txt  CMake build definition
└── makefile        convenience targets (make build, make install, make clean)
```

Studies and parameter sweeps live in a separate directory (e.g. `BioFEM-studies/`) and call `biofem` as an external command.

## Backends

| Target | Command | Hardware |
|---|---|---|
| DMPlex/MPI | `make build` | CPU, multi-node |
| Direct CUDA | `make build-cuda` | single GPU |
| Kokkos CUDA | `make build-kokkos-cuda` | GPU via Kokkos |
| Kokkos OpenMP | `make build-kokkos-openmp` | CPU via Kokkos |
