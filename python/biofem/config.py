"""Paths, solver backends, site config (config/site.cfg) and cluster detection."""
import configparser
import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — detect whether running from source tree or installed location
#
#   Source tree:  bin/biofem  →  ../CMakeLists.txt exists
#   Installed:    ~/.local/bin/biofem  →  no CMakeLists.txt above us
#
# _HERE is the directory of the bin/biofem launcher (it exports BIOFEM_BIN_DIR
# before importing this package); this package lives in <repo>/python/biofem.
# ---------------------------------------------------------------------------
_HERE      = Path(os.environ.get("BIOFEM_BIN_DIR", Path(__file__).resolve().parents[2] / "bin")).resolve()
_REPO_ROOT = _HERE.parent

_IN_SOURCE = (_REPO_ROOT / "CMakeLists.txt").exists()

if _IN_SOURCE:
    REPO_ROOT  = _REPO_ROOT
    BUILD_DIR  = REPO_ROOT / "build"
    CONFIG_DIR = REPO_ROOT / "config"
    # In the source tree executables may be named either way:
    #   cmake --build  → biofem-dmplex  (new)
    #   make dmplex    → testt-dmplex   (legacy)
    def _exe(cmake_name, legacy_name):
        p = BUILD_DIR / cmake_name
        return p if p.exists() else BUILD_DIR / legacy_name
else:
    REPO_ROOT  = None
    BUILD_DIR  = None
    CONFIG_DIR = _HERE.parent / "etc" / "biofem"   # optional installed config
    # Installed: executables are siblings in the same bin dir
    def _exe(cmake_name, _legacy):
        return _HERE / cmake_name

SITE_CFG = CONFIG_DIR / "site.cfg"

# load-env.sh lives at the repo root; for installed builds look one level up
# from the bin/ directory (i.e. <prefix>/load-env.sh).
if _IN_SOURCE:
    LOAD_ENV_SH = _REPO_ROOT / "load-env.sh"
else:
    _cand = _HERE.parent / "load-env.sh"
    LOAD_ENV_SH = _cand if _cand.exists() else None

BACKENDS = {
    "dmplex":        {"exe": _exe("biofem-dmplex",        "testt-dmplex"),        "gpu": False},
    "cuda":          {"exe": _exe("biofem-cuda",          "testt-cuda"),          "gpu": True},
    "kokkos-cuda":   {"exe": _exe("biofem-kokkos-cuda",   "testt-kokkos-cuda"),   "gpu": True},
    "kokkos-openmp": {"exe": _exe("biofem-kokkos-openmp", "testt-kokkos-openmp"), "gpu": False},
}

# ---------------------------------------------------------------------------
# Site config (config/site.cfg, gitignored, created by setup.sh)
# ---------------------------------------------------------------------------
def load_site_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "slurm":   {"account": "", "partition": ""},
        "modules": {"base": "", "petsc": "", "cuda": ""},
        "mpi":     {"extra_flags": ""},
    })
    if SITE_CFG.exists():
        cfg.read(SITE_CFG)
    return cfg


def _module_lines(cfg: configparser.ConfigParser, *, cuda: bool) -> str:
    mods = []
    for key in ("base", "petsc"):
        val = cfg.get("modules", key, fallback="").strip()
        if val:
            mods.extend(val.split())
    if cuda:
        val = cfg.get("modules", "cuda", fallback="").strip()
        if val:
            mods.extend(val.split())
    if not mods:
        return ""
    return "module purge\n" + "\n".join(f"ml {m}" for m in mods)


def _load_env_line() -> str:
    """Return a shell line that sources load-env.sh, or empty string if not found."""
    if LOAD_ENV_SH and LOAD_ENV_SH.exists():
        return f"source {LOAD_ENV_SH}"
    return ""


# ---------------------------------------------------------------------------
# Cluster detection
# ---------------------------------------------------------------------------
def detect_cluster() -> dict:
    return {
        "has_slurm":      shutil.which("sbatch") is not None,
        "in_allocation":  bool(os.environ.get("SLURM_JOB_ID")),
        "has_gpu":        (
            bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
            or shutil.which("nvidia-smi") is not None
        ),
    }
