# env/setup.sh
#
# Source this at the start of any work session on alpha01.
# It sets up paths and a few defaults. Equivalent to modules.sh on Polaris,
# but alpha01 has no Lmod, so this is mostly path manipulation.
#
# Usage:
#     source /mnt/nvme10/joseph_ufl/mlir-corpus-pipeline/env/setup.sh

# -----------------------------------------------------------------------------
# Project-level paths
# -----------------------------------------------------------------------------
export PROJECT_ROOT="/mnt/nvme10/joseph_ufl"
export LLVM_INSTALL="${PROJECT_ROOT}/llvm-cir"

# -----------------------------------------------------------------------------
# Make our custom LLVM the canonical clang/mlir-opt/llc on PATH (if installed)
# -----------------------------------------------------------------------------
if [ -d "${LLVM_INSTALL}/bin" ]; then
    export PATH="${LLVM_INSTALL}/bin:${PATH}"
    export LD_LIBRARY_PATH="${LLVM_INSTALL}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="${LLVM_INSTALL}/python_packages/mlir_core:${PYTHONPATH:-}"
fi

# -----------------------------------------------------------------------------
# Be a good neighbor on a shared box
# -----------------------------------------------------------------------------
# Cap default parallelism. Scripts that build things should respect this.
# 256 cores is too many; 32 is friendly. Override with `export NJOBS=N` if
# nobody else is using the machine.
export NJOBS="${NJOBS:-32}"

# Default niceness for long-running commands.
export NICE="${NICE:-nice -n 10}"

# -----------------------------------------------------------------------------
# Interactive sanity printout
# -----------------------------------------------------------------------------
if [ -t 1 ]; then
    echo "[setup.sh] PROJECT_ROOT = ${PROJECT_ROOT}"
    echo "[setup.sh] LLVM_INSTALL = ${LLVM_INSTALL}"
    echo "[setup.sh] NJOBS        = ${NJOBS}"
    if [ -x "${LLVM_INSTALL}/bin/clang" ]; then
        echo "[setup.sh] clang from: $(command -v clang)"
    else
        echo "[setup.sh] (custom LLVM not built yet — run scripts/build-llvm.sh)"
    fi
fi

# -----------------------------------------------------------------------------
# CUDA toolkit (headers + libraries only, no driver)
# -----------------------------------------------------------------------------
# Used by clang -fclangir --cuda-host-only for CUDA host-side lowering.
# The GPU driver is NOT installed on alpha01 and NOT required for host-only
# compilation. This is exploration/lowering mode; no binaries will run here.
export CUDA_HOME="${PROJECT_ROOT}/cuda-12.6.3"
if [ -d "${CUDA_HOME}/bin" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
