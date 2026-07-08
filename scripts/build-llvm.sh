#!/bin/bash
# =============================================================================
# scripts/build-llvm.sh
# =============================================================================
# Build LLVM/Clang with ClangIR enabled. Runs directly (no scheduler).
# Use tmux so disconnects don't kill the build:
#
#     tmux new -s llvm-build
#     bash scripts/build-llvm.sh 2>&1 | tee build-llvm.out
#     # Ctrl-B, D to detach. Reattach with `tmux attach -t llvm-build`.
#
# Expected duration on this 256-core, 3-TiB box: 12-20 minutes.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Working directory: same as the script's containing repo.
# -----------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${REPO_ROOT}"

# Load shared env (paths, NJOBS, NICE).
source env/setup.sh

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
LLVM_SRC="${PROJECT_ROOT}/llvm-project-src"
# alpha01 has no node-local scratch concept; /tmp is fine but might be tmpfs
# (RAM-backed). On this 3-TiB box that's actually a feature — the build is
# fast and tmpfs is the fastest possible scratch. If /tmp gets full, switch
# to ${PROJECT_ROOT}/llvm-build (uses NVMe, still fast).
BUILD_DIR="/tmp/llvm-build-$$"

# -----------------------------------------------------------------------------
# Diagnostic header
# -----------------------------------------------------------------------------
echo "===================================================================="
echo "LLVM-with-ClangIR build (alpha01)"
echo "===================================================================="
echo "Date:          $(date)"
echo "Host:          $(hostname)"
echo "PROJECT_ROOT:  ${PROJECT_ROOT}"
echo "LLVM_SRC:      ${LLVM_SRC}"
echo "BUILD_DIR:     ${BUILD_DIR}"
echo "LLVM_INSTALL:  ${LLVM_INSTALL}"
echo "Parallel jobs: ${NJOBS}"
echo "Load avg:      $(uptime | awk -F'load average:' '{print $2}')"
echo "===================================================================="

# -----------------------------------------------------------------------------
# Step 1: Get the LLVM source
# -----------------------------------------------------------------------------
LLVM_COMMIT="HEAD"  # TODO: pin a 40-char SHA after first successful build

if [ -d "${LLVM_SRC}/.git" ]; then
    echo "[1/4] LLVM source already cloned, fetching latest..."
    cd "${LLVM_SRC}"
    git fetch origin
    git checkout "${LLVM_COMMIT}"
else
    echo "[1/4] Cloning LLVM source (a few minutes)..."
    # --depth=1 skips ~5 GB of git history.
    git clone --depth=1 https://github.com/llvm/llvm-project.git "${LLVM_SRC}"
    cd "${LLVM_SRC}"
fi
echo "    HEAD: $(git rev-parse HEAD)"

# -----------------------------------------------------------------------------
# Step 2: Configure
# -----------------------------------------------------------------------------
echo "[2/4] Configuring build in ${BUILD_DIR}..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Same flags as Polaris build, with one difference: NVPTX target is dropped
# (no NVIDIA GPUs here). host (X86) is sufficient.
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DCLANG_ENABLE_CIR=ON \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_PARALLEL_LINK_JOBS=8 \
    -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL}" \
    "${LLVM_SRC}/llvm"

# -----------------------------------------------------------------------------
# Step 3: Build (the long step)
# -----------------------------------------------------------------------------
echo "[3/4] Building with ${NJOBS} parallel jobs (niced)..."
time ${NICE} ninja -j "${NJOBS}"

# -----------------------------------------------------------------------------
# Step 4: Install
# -----------------------------------------------------------------------------
echo "[4/4] Installing to ${LLVM_INSTALL}..."
rm -rf "${LLVM_INSTALL}"
mkdir -p "${LLVM_INSTALL}"
ninja install

# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
echo "===================================================================="
echo "Smoke testing the install..."
echo "===================================================================="
INSTALLED_CLANG="${LLVM_INSTALL}/bin/clang"

"${INSTALLED_CLANG}" --version

TEST_C="/tmp/cir_smoketest_$$.c"
TEST_CIR="/tmp/cir_smoketest_$$.cir"
cat > "${TEST_C}" <<'TESTEOF'
int add(int a, int b) { return a + b; }
TESTEOF

"${INSTALLED_CLANG}" -fclangir -Xclang -emit-cir -S "${TEST_C}" -o "${TEST_CIR}"

if grep -q "cir.func" "${TEST_CIR}"; then
    echo "    PASS: emitted CIR contains cir.func"
    head -20 "${TEST_CIR}"
else
    echo "    FAIL: emitted file does not contain cir.func"
    cat "${TEST_CIR}"
    exit 1
fi

"${LLVM_INSTALL}/bin/mlir-opt" --version | head -5

# Cleanup
rm -f "${TEST_C}" "${TEST_CIR}"
rm -rf "${BUILD_DIR}"

echo "===================================================================="
echo "Done. LLVM with ClangIR installed at:"
echo "    ${LLVM_INSTALL}"
echo "Use it by sourcing env/setup.sh in your shell."
echo "===================================================================="
