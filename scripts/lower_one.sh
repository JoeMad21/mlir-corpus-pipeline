#!/bin/bash
# scripts/lower_one.sh (v2)
# Changes: source auto-detect handles src/main.cu layouts and depth-2 searches;
# compile failures now extract real clang diagnostics into compile-errors.txt.

set -uo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "usage: $0 <benchmark_source_dir> [output_base_dir]" >&2
    exit 1
fi

BENCH_DIR="$1"
OUTPUT_BASE="${2:-/mnt/nvme10/joseph_ufl/corpus-alpha01}"

if [ ! -d "${BENCH_DIR}" ]; then
    echo "error: benchmark directory does not exist: ${BENCH_DIR}" >&2
    exit 1
fi
BENCH_NAME="$(basename "${BENCH_DIR}")"

if ! command -v clang > /dev/null 2>&1; then
    echo "error: clang not on PATH. Source env/setup.sh first." >&2
    exit 1
fi
if [ -z "${CUDA_HOME:-}" ] || [ ! -d "${CUDA_HOME}" ]; then
    echo "error: CUDA_HOME not set. Source env/setup.sh." >&2
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
PARSER="${REPO_ROOT}/tools/parse_snapshots.py"

if [ ! -f "${PARSER}" ]; then
    echo "error: parser not found at ${PARSER}" >&2
    exit 1
fi

# Source detection: prefer main.cu at top, then src/main.cu, then any main.cu
# within depth 2, then any single .cu file.
SOURCE=""
if   [ -f "${BENCH_DIR}/main.cu" ];       then SOURCE="${BENCH_DIR}/main.cu"
elif [ -f "${BENCH_DIR}/src/main.cu" ];   then SOURCE="${BENCH_DIR}/src/main.cu"
else
    MAIN_CANDIDATES=$(find "${BENCH_DIR}" -maxdepth 2 -name 'main.cu' -not -name '*.cuh' 2>/dev/null)
    if [ -n "${MAIN_CANDIDATES}" ]; then
        SOURCE=$(echo "${MAIN_CANDIDATES}" | head -1)
    else
        CU_FILES=$(find "${BENCH_DIR}" -maxdepth 2 -name '*.cu' -not -name '*.cuh' 2>/dev/null)
        CU_COUNT=$(echo "${CU_FILES}" | grep -c . || true)
        if [ "${CU_COUNT}" -eq 1 ]; then
            SOURCE="${CU_FILES}"
        elif [ "${CU_COUNT}" -gt 1 ]; then
            echo "warning: multiple .cu files in ${BENCH_NAME}, picking first" >&2
            SOURCE=$(echo "${CU_FILES}" | head -1)
        fi
    fi
fi

if [ -z "${SOURCE}" ] || [ ! -f "${SOURCE}" ]; then
    echo "error: could not auto-detect source file in ${BENCH_DIR}" >&2
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_DIR="${OUTPUT_BASE}/${BENCH_NAME}-${TIMESTAMP}"
mkdir -p "${OUT_DIR}/snapshots"

exec > >(tee "${OUT_DIR}/driver.log") 2>&1

echo "===================================================================="
echo "lower_one.sh (v2)"
echo "===================================================================="
echo "Benchmark:  ${BENCH_NAME}"
echo "Source:     ${SOURCE}"
echo "Output:     ${OUT_DIR}"
echo "Timestamp:  ${TIMESTAMP}"
echo "===================================================================="

SOURCE_EXT="${SOURCE##*.}"
cp "${SOURCE}" "${OUT_DIR}/source.${SOURCE_EXT}"

echo
echo "[1/3] Compiling with ClangIR + snapshot capture..."

BENCH_PARENT="$(dirname "${BENCH_DIR}")"

# Compute sibling-directory paths. HeCBench often uses the SYCL/HIP/OMP
# variants of a benchmark as the source of shared utility headers.
BENCH_STEM="$(basename "${BENCH_DIR}" -cuda)"
SIBLING_SYCL="${BENCH_PARENT}/${BENCH_STEM}-sycl"
SIBLING_HIP="${BENCH_PARENT}/${BENCH_STEM}-hip"
SIBLING_OMP="${BENCH_PARENT}/${BENCH_STEM}-omp"
CLANG_CMD=(
    clang
    --cuda-host-only
    --cuda-path="${CUDA_HOME}"
    -fclangir
    -I "${BENCH_DIR}"
    -I "${BENCH_DIR}/src"
    -I "${BENCH_PARENT}"
    -I "${BENCH_DIR}/include"
    -I "${BENCH_PARENT}/include"
    -I "${SIBLING_SYCL}"
    -I "${SIBLING_HIP}"
    -I "${SIBLING_OMP}"
    -DNVTX_DISABLE
    -Ddfloat=float
    -Ddlong=int
    -Xclang -mmlir -Xclang --mlir-print-ir-after-all
    -Xclang -mmlir -Xclang --mlir-print-ir-after-change
    -Xclang -mmlir -Xclang --mlir-print-ir-module-scope
    -Xclang -mmlir -Xclang --mlir-disable-threading
    -S "${SOURCE}"
    -o "${OUT_DIR}/main.ll"
)
echo "    ${CLANG_CMD[*]}"

"${CLANG_CMD[@]}" > /dev/null 2> "${OUT_DIR}/snapshot-stream.txt"
CLANG_RC=$?

STREAM_BYTES=$(stat -c%s "${OUT_DIR}/snapshot-stream.txt" 2>/dev/null || echo 0)
LL_BYTES=$(stat -c%s "${OUT_DIR}/main.ll" 2>/dev/null || echo 0)
echo "    clang exit=${CLANG_RC}, stream=${STREAM_BYTES}B, main.ll=${LL_BYTES}B"

if [ "${CLANG_RC}" -ne 0 ]; then
    # Extract real clang diagnostics, not IR-dump tails.
    grep -E '(error|fatal error|warning): |errors generated' \
        "${OUT_DIR}/snapshot-stream.txt" > "${OUT_DIR}/compile-errors.txt" \
        2>/dev/null || true
    grep -E '^In file included from ' \
        "${OUT_DIR}/snapshot-stream.txt" >> "${OUT_DIR}/compile-errors.txt" \
        2>/dev/null || true

    ERR_LINES=$(wc -l < "${OUT_DIR}/compile-errors.txt" 2>/dev/null || echo 0)
    echo "    FAIL: clang returned nonzero. Extracted ${ERR_LINES} diagnostic lines."
    echo "    See ${OUT_DIR}/compile-errors.txt"
    if [ "${ERR_LINES}" -gt 0 ]; then
        echo "    First 10 diagnostics:"
        head -10 "${OUT_DIR}/compile-errors.txt" | sed 's/^/      /'
    else
        echo "    (No matching diagnostics found. Last 20 raw lines:)"
        tail -20 "${OUT_DIR}/snapshot-stream.txt" | sed 's/^/      /'
    fi
    exit 2
fi

echo
echo "[2/3] Parsing snapshot stream..."
python3 "${PARSER}" \
    --stream "${OUT_DIR}/snapshot-stream.txt" \
    --outdir "${OUT_DIR}/snapshots" \
    --source "${SOURCE}"
PARSE_RC=$?

SNAP_COUNT=$(find "${OUT_DIR}/snapshots" -maxdepth 1 -name '*.mlir' | wc -l)
echo "    parser exit=${PARSE_RC}, snapshots=${SNAP_COUNT}"

if [ "${PARSE_RC}" -ne 0 ] || [ "${SNAP_COUNT}" -eq 0 ]; then
    echo "    FAIL: parser produced no snapshots."
    exit 3
fi

echo
echo "[3/3] Round-tripping snapshots through cir-opt (Policy B: filter, don't fail)..."
KEPT_FILES=()
DROPPED_FILES=()
: > "${OUT_DIR}/roundtrip.log"

for snap in "${OUT_DIR}/snapshots"/*.mlir; do
    name=$(basename "${snap}")
    if cir-opt "${snap}" > /dev/null 2> "${OUT_DIR}/roundtrip.log.tmp"; then
        echo "    OK   ${name}"
        echo "OK   ${name}" >> "${OUT_DIR}/roundtrip.log"
        KEPT_FILES+=("${name}")
    else
        rc=$?
        echo "    DROP ${name} (roundtrip failed, exit ${rc})"
        echo "DROP ${name} (roundtrip failed, exit ${rc})" >> "${OUT_DIR}/roundtrip.log"
        echo "----- first 10 lines of stderr -----" >> "${OUT_DIR}/roundtrip.log"
        head -10 "${OUT_DIR}/roundtrip.log.tmp" >> "${OUT_DIR}/roundtrip.log"
        # Policy B: physically remove the bad snapshot so it can't leak into corpus.
        rm -f "${snap}"
        DROPPED_FILES+=("${name}")
    fi
done
rm -f "${OUT_DIR}/roundtrip.log.tmp"

KEPT_COUNT=${#KEPT_FILES[@]}
DROPPED_COUNT=${#DROPPED_FILES[@]}

# Update manifest.json: annotate each snapshot with roundtrip_status and add
# aggregate counts at the module level. Uses Python (already required for
# parse_snapshots.py) so no new dependencies.
KEPT_CSV=$(IFS=,; echo "${KEPT_FILES[*]}")
python3 - "${OUT_DIR}/snapshots/manifest.json" "${KEPT_CSV}" <<'PYEOF'
import json, sys
manifest_path = sys.argv[1]
kept = set(filter(None, sys.argv[2].split(',')))
with open(manifest_path) as f:
    m = json.load(f)
for snap in m['snapshots']:
    snap['roundtrip_status'] = 'ok' if snap['filename'] in kept else 'failed'
m['snapshots_kept']    = sum(1 for s in m['snapshots'] if s['roundtrip_status'] == 'ok')
m['snapshots_dropped'] = sum(1 for s in m['snapshots'] if s['roundtrip_status'] == 'failed')
with open(manifest_path, 'w') as f:
    json.dump(m, f, indent=2)
    f.write('\n')
PYEOF

echo
echo "===================================================================="
if [ "${DROPPED_COUNT}" -eq 0 ]; then
    echo "SUCCESS: ${SNAP_COUNT} snapshots, all round-trip through cir-opt."
    echo "Corpus entry:  ${OUT_DIR}"
    echo "===================================================================="
    exit 0
elif [ "${KEPT_COUNT}" -eq 0 ]; then
    echo "FAILURE: 0 of ${SNAP_COUNT} snapshots survived roundtrip."
    echo "See ${OUT_DIR}/roundtrip.log for details."
    echo "===================================================================="
    exit 4
else
    echo "PARTIAL_KEPT: ${KEPT_COUNT} of ${SNAP_COUNT} snapshots retained; ${DROPPED_COUNT} dropped."
    echo "Corpus entry:  ${OUT_DIR}"
    echo "See ${OUT_DIR}/roundtrip.log for dropped-snapshot details."
    echo "===================================================================="
    exit 5
fi
