# mlir-corpus-pipeline (alpha01)

This is the alpha01 mirror of the mlir-corpus-pipeline project. It has the
same goal as the Polaris instance (generate MLIR/LLVM IR snapshots from
benchmarks for LLM training corpus) but adapted for:

- A single shared bare-metal node (no PBS, no modules)
- User-level installs only
- CPU-only validation (no NVIDIA GPUs available here)
- Tenstorrent and Furiosa accelerators available for future Phase 3 work

## Differences from the Polaris instance

| Concern         | Polaris                       | alpha01                          |
|-----------------|-------------------------------|----------------------------------|
| Scheduler       | PBS Pro (qsub)                | None (bash + tmux)               |
| Modules         | Lmod                          | Direct system tools              |
| Build job       | scripts/build-llvm.pbs        | scripts/build-llvm.sh            |
| Env setup       | env/modules.sh                | env/setup.sh                     |
| Validation tgt  | NVIDIA A100 (CUDA)            | EPYC 9534 (OpenMP)               |
| Benchmark suite | HeCBench *-cuda variants      | HeCBench *-omp variants          |
| Scratch space   | /local/scratch (ephemeral)    | /tmp (use sparingly, 256 GB RAM) |

## Layout

    /mnt/nvme10/joseph_ufl/
    |-- mlir-corpus-pipeline/     # this project (scripts and source)
    |-- llvm-project-src/         # LLVM source
    |-- llvm-cir/                 # LLVM install with ClangIR enabled
    |-- HeCBench/                 # benchmark suite
    `-- corpus-alpha01/           # generated IR snapshots from this node

## Quickstart

    source env/setup.sh
    bash scripts/build-llvm.sh    # one-time, ~12-15 min on this box

## Etiquette on a shared machine

- Cap parallel build jobs to ~half the cores (`-j 32` or `-j 48`, not 256).
- Use `nice -n 10` on long-running builds so others' jobs are prioritized.
- Run inside `tmux` so disconnects don't kill work and others can see what's
  active.
