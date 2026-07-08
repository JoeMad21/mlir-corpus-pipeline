#!/bin/bash
# scripts/lower_all.sh (v2)
# Changes: first_error column now pulls from compile-errors.txt or
# roundtrip.log (whichever fits the failure mode). Live output shows a
# truncated error snippet for failures.

set -uo pipefail

SRC_DIR="/mnt/nvme10/joseph_ufl/HeCBench/src"
PATTERN="*-cuda"
LIMIT=20
JOBS=8
TIMEOUT_SEC=60
OUTPUT_BASE="/mnt/nvme10/joseph_ufl/corpus-alpha01"
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --src)      SRC_DIR="$2"; shift 2 ;;
        --pattern)  PATTERN="$2"; shift 2 ;;
        --limit)    LIMIT="$2"; shift 2 ;;
        --jobs)     JOBS="$2"; shift 2 ;;
        --timeout)  TIMEOUT_SEC="$2"; shift 2 ;;
        --output)   OUTPUT_BASE="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  sed -n '/^# Usage:/,/^set /p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *)          echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOWER_ONE="${SCRIPT_DIR}/lower_one.sh"

if [ ! -x "${LOWER_ONE}" ]; then
    echo "error: lower_one.sh not found or not executable at ${LOWER_ONE}" >&2
    exit 1
fi
if [ ! -d "${SRC_DIR}" ]; then
    echo "error: --src does not exist: ${SRC_DIR}" >&2
    exit 1
fi
if ! command -v clang > /dev/null 2>&1; then
    echo "error: clang not on PATH. Source env/setup.sh first." >&2
    exit 1
fi

mapfile -t ALL_BENCHES < <(
    find "${SRC_DIR}" -maxdepth 1 -mindepth 1 -type d -name "${PATTERN}" | sort
)
TOTAL_FOUND=${#ALL_BENCHES[@]}
if [ "${LIMIT}" -gt 0 ] && [ "${TOTAL_FOUND}" -gt "${LIMIT}" ]; then
    BENCHES=("${ALL_BENCHES[@]:0:${LIMIT}}")
else
    BENCHES=("${ALL_BENCHES[@]}")
fi
TOTAL_SELECTED=${#BENCHES[@]}

if [ "${TOTAL_SELECTED}" -eq 0 ]; then
    echo "error: no benchmarks matched '${PATTERN}' under ${SRC_DIR}" >&2
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BATCH_DIR="${OUTPUT_BASE}/batch-runs/batch-${TIMESTAMP}"
mkdir -p "${BATCH_DIR}/failures" "${BATCH_DIR}/logs"

SUMMARY="${BATCH_DIR}/summary.csv"
BATCH_LOG="${BATCH_DIR}/batch.log"

echo "benchmark,exit_code,status,snapshots,wall_seconds,first_error" > "${SUMMARY}"

exec > >(tee "${BATCH_LOG}") 2>&1

echo "===================================================================="
echo "lower_all.sh (v2)"
echo "===================================================================="
echo "SRC_DIR:        ${SRC_DIR}"
echo "Pattern:        ${PATTERN}"
echo "Total found:    ${TOTAL_FOUND}"
echo "Limit:          ${LIMIT} (running ${TOTAL_SELECTED})"
echo "Concurrency:    ${JOBS}"
echo "Timeout:        ${TIMEOUT_SEC}s per benchmark"
echo "Batch dir:      ${BATCH_DIR}"
echo "===================================================================="

if [ "${DRY_RUN}" -eq 1 ]; then
    echo
    echo "DRY RUN - would process:"
    for b in "${BENCHES[@]}"; do
        echo "  $(basename "$b")"
    done
    exit 0
fi

csv_clean() {
    local s="$1"
    s="${s//,/;}"
    s="${s//$'\n'/ }"
    s="${s//$'\r'/ }"
    s="${s//\"/}"
    printf '%.200s' "${s}"
}

run_one() {
    local bench_path="$1"
    local bench_name
    bench_name="$(basename "${bench_path}")"

    local start_epoch
    start_epoch=$(date +%s)
    local per_log="${BATCH_DIR}/logs/${bench_name}.log"

    timeout "${TIMEOUT_SEC}s" \
        "${LOWER_ONE}" "${bench_path}" "${OUTPUT_BASE}" \
        > "${per_log}" 2>&1
    local rc=$?

    local end_epoch
    end_epoch=$(date +%s)
    local wall=$((end_epoch - start_epoch))

    local status
    case "${rc}" in
        0)   status="success" ;;
        1)   status="driver_error" ;;
        2)   status="compile_fail" ;;
        3)   status="parse_fail" ;;
        4)   status="roundtrip_fail" ;;
        5)   status="partial_kept" ;;
        124) status="timeout" ;;
        137) status="killed" ;;
        *)   status="unknown_${rc}" ;;
    esac

    local snap_count=0
    if [ "${rc}" -eq 0 ] || [ "${rc}" -eq 4 ] || [ "${rc}" -eq 5 ]; then
        snap_count=$(grep -oE '[0-9]+ snapshots' "${per_log}" | head -1 | awk '{print $1}')
        snap_count="${snap_count:-0}"
    fi

    local latest=""
    latest=$(find "${OUTPUT_BASE}" -maxdepth 1 -type d -name "${bench_name}-*" \
                -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')

    local first_err=""
    case "${status}" in
        compile_fail)
            if [ -n "${latest}" ] && [ -s "${latest}/compile-errors.txt" ]; then
                first_err=$(head -1 "${latest}/compile-errors.txt")
            fi
            ;;
        roundtrip_fail)
            if [ -n "${latest}" ] && [ -f "${latest}/roundtrip.log" ]; then
                first_err=$(grep -m1 '^FAIL' "${latest}/roundtrip.log")
            fi
            ;;
        driver_error)
            first_err=$(grep -m1 '^error:' "${per_log}")
            ;;
        timeout|killed)
            first_err="killed after ${TIMEOUT_SEC}s"
            ;;
        parse_fail)
            first_err=$(grep -m1 -E 'parser|error' "${per_log}")
            ;;
    esac

    if [ -z "${first_err}" ] && [ "${rc}" -ne 0 ]; then
        first_err=$(grep -m1 -E '^(error:|.*error:)' "${per_log}" | head -1)
    fi

    first_err=$(csv_clean "${first_err}")

    if [ "${rc}" -ne 0 ] && [ "${rc}" -ne 1 ] && [ -n "${latest}" ]; then
        ln -sfn "${latest}" "${BATCH_DIR}/failures/${bench_name}"
    fi

    (
        flock -x 200
        echo "${bench_name},${rc},${status},${snap_count},${wall},${first_err}" >> "${SUMMARY}"
    ) 200>>"${SUMMARY}.lock"

    if [ "${rc}" -eq 0 ]; then
        printf "  [%s] %-40s rc=%d (%ds)\n" \
            "${status}" "${bench_name}" "${rc}" "${wall}"
    else
        local err_snip="${first_err:0:80}"
        printf "  [%s] %-40s rc=%d (%ds) %s\n" \
            "${status}" "${bench_name}" "${rc}" "${wall}" "${err_snip}"
    fi
}

echo
echo "Starting batch..."
echo

for bench in "${BENCHES[@]}"; do
    while [ "$(jobs -r | wc -l)" -ge "${JOBS}" ]; do
        wait -n 2>/dev/null || true
    done
    run_one "${bench}" &
done
wait
rm -f "${SUMMARY}.lock"

echo
echo "===================================================================="
echo "Batch complete."
echo "===================================================================="
echo "Status breakdown:"
tail -n +2 "${SUMMARY}" | awk -F, '{print $3}' | sort | uniq -c | sort -rn \
    | awk '{printf "  %5d  %s\n", $1, $2}'
echo
echo "Summary CSV:    ${SUMMARY}"
echo "Batch log:      ${BATCH_LOG}"
echo "Failure links:  ${BATCH_DIR}/failures/"
echo "===================================================================="

# Failure-mode breakdown: group non-success rows by (status, first_err trimmed)
echo
echo "Failure mode breakdown (top 15):"
tail -n +2 "${SUMMARY}" | awk -F, '$3 != "success"' | \
    awk -F, '{ printf "%s: %.60s\n", $3, $6 }' | \
    sort | uniq -c | sort -rn | head -15 | \
    awk '{ n=$1; $1=""; sub(/^ /,""); printf "  %5d  %s\n", n, $0 }'
echo

FAIL_COUNT=$(tail -n +2 "${SUMMARY}" | awk -F, '$3 != "success"' | wc -l)
if [ "${FAIL_COUNT}" -eq 0 ]; then
    exit 0
else
    echo "WARN: ${FAIL_COUNT} benchmarks did not complete successfully." >&2
    exit 2
fi
