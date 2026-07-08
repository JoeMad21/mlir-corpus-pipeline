# mlir-corpus-pipeline

A pipeline for generating a corpus of MLIR intermediate representations from
HPC and ML benchmarks, intended as training data for compiler-focused
language models. Source programs are lowered through the ClangIR compiler
stack and the IR is captured at every pass boundary where it changes.

The initial target corpus is the [HeCBench](https://github.com/zjin-lcf/HeCBench)
benchmark suite. On the reference deployment (a single shared bare-metal
server with no NVIDIA GPU), 100 CUDA benchmarks yield ~90% effective
retention: 71 clean successes, 19 partial-kept, 10 failures.

## What this is not

Not yet a Polaris/HPC-scheduler variant.
Not yet handling Fortran (via Flang) or Python (via torch-mlir/xDSL). Those
are planned; this repo is the working v0 for CUDA-host-side lowering.

Not yet correctness-gated against reference binaries. Alpha01 has no GPU
so we cannot run the compiled binaries; we operate in "lower-only" mode
and validate only that snapshots are consumable by cir-opt.

## Prerequisites

* Linux (developed on Ubuntu 22.04)
* GCC 10+ or Clang, CMake 3.20+, Ninja, Git, Python 3.10+
* ~15 GB free disk
* Network access (proxy is fine)
* User-level access; root/sudo not required

## One-time setup

    # 1. Clone into a writable workspace
    cd /path/with/space
    git clone https://github.com/JoeMad21/mlir-corpus-pipeline.git
    cd mlir-corpus-pipeline

    # 2. Edit env/setup.sh to set PROJECT_ROOT to your workspace path

    # 3. Install nanobind (MLIR Python bindings dependency)
    python3 -m pip install --user nanobind

    # 4. Build LLVM with ClangIR enabled (15-45 min depending on cores)
    nohup bash scripts/build-llvm.sh > build-llvm.out 2>&1 < /dev/null &

    # 5. Install CUDA toolkit to $CUDA_HOME (headers/libs only, no driver)
    #    See scripts/build-llvm.sh comments for the runfile invocation.

    # 6. Clone HeCBench alongside this repo
    cd $PROJECT_ROOT
    git clone --depth=1 https://github.com/zjin-lcf/HeCBench.git

## Running the pipeline

Every session, source the environment:

    source env/setup.sh

Lower a single benchmark:

    bash scripts/lower_one.sh $PROJECT_ROOT/HeCBench/src/reverse-cuda

Output goes to `$PROJECT_ROOT/corpus-alpha01/reverse-cuda-<timestamp>/`
containing the source copy, LLVM IR, snapshot files, and a manifest.json.

Batch over many benchmarks:

    bash scripts/lower_all.sh --limit 20
    bash scripts/lower_all.sh --limit 0     # all ~529 CUDA benchmarks

Output goes to `$PROJECT_ROOT/corpus-alpha01/batch-runs/batch-<timestamp>/`
with a summary.csv and per-benchmark logs.

## Exit codes

`lower_one.sh` returns:

    0  All snapshots kept and round-trip through cir-opt cleanly
    1  Invalid arguments or missing tools
    2  Clang compile failed
    3  Parser produced zero snapshots
    4  All snapshots failed roundtrip validation
    5  Some snapshots kept, some dropped (Policy B: partial data still useful)

`lower_all.sh` maps these into status strings in summary.csv:
`success`, `driver_error`, `compile_fail`, `parse_fail`, `roundtrip_fail`,
`partial_kept`, `timeout`, `killed`, `unknown_N`.

## What the pipeline does

For each benchmark, `lower_one.sh`:

1. Auto-detects the source file (handles main.cu, src/main.cu, and other
   HeCBench layouts).
2. Runs `clang -fclangir --cuda-host-only` with IR-dump instrumentation
   to capture the MLIR module after every pass that changes it.
3. Parses the raw dump stream into one .mlir file per pass, plus a
   manifest.json describing the ordered chain with parent-child hashes.
4. Round-trips each snapshot through `cir-opt` as a validity check.
   Snapshots that fail to re-parse are physically removed from disk and
   marked in the manifest as `roundtrip_status: "failed"` (Policy B).

Result: 3-7 valid MLIR snapshots per benchmark, forming an ordered chain
from initial CIR down to LLVM dialect.

## Flags baked into the pipeline

The `clang` invocation in `lower_one.sh` includes several flags whose
rationale is captured here. Do not remove these without understanding why
they are present.

    --cuda-host-only              Skip device code; we have no GPU here.
    --cuda-path=$CUDA_HOME        Find CUDA headers.
    -fclangir                     Enable ClangIR (the keystone flag).

Include paths (clang silently ignores nonexistent ones, so we always
add all of them):

    -I $BENCH_DIR                 Benchmark's top-level dir.
    -I $BENCH_DIR/src             Some benchmarks nest sources.
    -I $BENCH_DIR/include         Per-benchmark include dir (ans-cuda, etc.).
    -I $BENCH_PARENT              HeCBench src/ dir.
    -I $BENCH_PARENT/include      HeCBench shared headers (SDKBitMap.h).
    -I $SIBLING_SYCL              SYCL variant, source of shared utils.
    -I $SIBLING_HIP               HIP variant, ditto.
    -I $SIBLING_OMP               OpenMP variant, ditto.

Compile-time defines:

    -DNVTX_DISABLE                NVTX headers use atomic builtins that
                                  ClangIR has not implemented yet. This
                                  disables them; recovers ~20% of the
                                  benchmarks that would otherwise fail.
    -Ddfloat=float                HeCBench uses these as configurable
    -Ddlong=int                   precision/index types set in Makefiles.
                                  We force float/int to make the benchmarks
                                  compile even when the Makefile said double.

IR dump instrumentation (must be forwarded via -Xclang -mmlir -Xclang):

    --mlir-print-ir-after-all     Dump the module after every pass.
    --mlir-print-ir-after-change  Only dump when the pass changed IR.
    --mlir-print-ir-module-scope  Print whole module, not just op scope.
    --mlir-disable-threading      Deterministic pass ordering.

The --after-all and --after-change flags MUST be combined; --after-change
alone is silently ignored by clang's pass manager.

## What actually gets captured

Observed on a 100-benchmark HeCBench-CUDA run:

    Status breakdown:
       71  success
       19  partial_kept
        9  compile_fail
        1  driver_error

    Snapshot count per benchmark: 3-7 (avg 5.16)
    Total snapshots captured:     516
    Corpus size on disk:          ~450 MB

    Passes observed (frequency across corpus):
      universal (~100%):  cir-canonicalize, cir-cxxabi-lowering,
                          cir-flat-to-llvm, cir-hoist-allocas
      very common (85%):  cir-flatten-cfg
      half (~50%):        cir-lowering-prepare
      selective (5-15%):  cir-eh-abi-lowering, cir-goto-solver
      rare (<1%):         cir-target-lowering, omp-mark-declare-target

## Known failure modes

Handled automatically:

* Some ClangIR-emitted snapshots fail cir-opt roundtrip (clang's printer
  emits syntax its own parser rejects, most commonly on `cir.alloca` with
  cleanup_dest_slot markers). Policy B removes these; the rest of the
  benchmark's snapshots still contribute.

Not yet handled:

* Benchmarks that use MPI, NCCL, QuantLib, or other external libraries not
  present on the machine. Currently fail with compile_fail.
* Two benchmarks (bm3d-cuda, bn-cuda) produce ~15 MB snapshots because of
  template instantiation via OpenCV/Thrust. Handled fine by the current
  pipeline but will need special treatment during Parquet aggregation.
* Benchmarks with source layouts we don't recognize (e.g. all sources in
  deep subdirectories) fail with driver_error. About 1% of the suite.

## Directory layout

    $PROJECT_ROOT/
        mlir-corpus-pipeline/         # this repo
        llvm-project-src/             # LLVM source (created by build script)
        llvm-cir/                     # LLVM install (~6 GB)
        cuda-12.6.3/                  # CUDA toolkit (~7 GB)
        HeCBench/                     # benchmark source
        corpus-alpha01/               # generated snapshots

## Repository layout

    env/setup.sh                      # Environment loader
    scripts/build-llvm.sh             # One-time LLVM+ClangIR build
    scripts/lower_one.sh              # Per-benchmark driver
    scripts/lower_all.sh              # Batch runner
    tools/parse_snapshots.py          # IR-dump stream parser
    tools/build_parquet.py            # Parquet aggregation of the corpus

## Roadmap

Near-term:

* Full HeCBench sweep (~529 benchmarks) with longer timeout for whales

Medium-term:

* Fortran frontend via Flang (`-emit-mlir`)
* Python frontends via torch-mlir and xDSL
* Correctness gate on a GPU-equipped machine (Polaris variant)
* Handling of benchmarks with nonstandard source layouts

Longer-term:

* Aggregate corpus across multiple source languages, deduplicate by IR hash
* Publish as a HuggingFace dataset
