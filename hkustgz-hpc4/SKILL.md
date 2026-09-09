---
name: hkustgz-hpc4
description: Operate the HKUST-GZ HPC Phase 4 domestic platform through CPU/NPU Slurm jobs. Use for HPC4/四期, hkustgz-hpc4, hpc4login.hpc.hkust-gz.edu.cn, Kunpeng CPU, and 910C/A3 NPU work. If this is the only installed HKUST-GZ HPC skill, use it for otherwise-unspecified HKUST-GZ cluster work.
metadata:
  short-description: CPU/NPU Slurm workflows on HKUST-GZ HPC Phase 4
---

# HKUST-GZ HPC4

## Choose the execution path from live evidence

**HPC4 supports CPU and NPU jobs through Slurm. A container is not required for the verified SSH/Slurm environment.** The scheduler allocates resources; the Python environment supplies software.

Official documentation: https://docs.hpc.hkust-gz.edu.cn/docs/hpc4/domestic/

Documentation may describe a different deployment or lag the live cluster. Missing Slurm NPU instructions do not mean NPU Slurm is unavailable. Prefer current scheduler output and verified job behavior for the selected endpoint; do not transfer Phase 2 partition names, CUDA directives, or storage assumptions.

Keep an already established phase/SSH target. If both Phase 2 and Phase 4 skills are installed, inspect the project and SSH context before asking which phase. Use `hkustgz-hpc2` for Phase 2/A800/A40; Phase 1 uses LSF.

## Connect and inspect

Use the configured `hkustgz-hpc4` SSH alias when available, otherwise:

```bash
ssh <username>@hpc4login.hpc.hkust-gz.edu.cn
```

Portal: https://hpc4login.hpc.hkust-gz.edu.cn/#/app/user

Use the login node for file editing, scheduler inspection, and submission. Run training, evaluation, and compute benchmarks inside an allocation.

```bash
sinfo -h -o "%P %G"
scontrol show partition
squeue -u "$USER"
```

Observed on this SSH endpoint, rechecked **2026-09-10**:

- CPU partitions: `a128m512u` (default), `a128m512ue`, `emergency`, `debug`.
- NPU partitions: `a320m2tn910cu`, `a320m2tn910cue`, `emergency_a320m2tn910c`; NPU nodes advertise `npu:16`.
- Use the ordinary partition appropriate to the task; do not assume emergency/debug partition eligibility.
- The older documentation's CPU partition `hpc` is not present in this inventory. Discover partitions before submission instead of hard-coding it.

These are dated observations, not permanent cluster guarantees. Query account/QOS limits and current partition availability when choosing resources.

## Resources and authorization

Before submission, make the actual resource request reviewable: partition, account if needed, walltime, nodes, tasks, CPUs, host memory, NPU chips and physical-card equivalent, job name, working directory, output/error paths, environment, and command.

Proceed within the user's existing authorization, including explicitly authorized execution or bounded optimization. Ask only when resource consumption or scope is not already authorized; do not require repeated approval merely because another submission is needed within that scope. `sbatch --test-only` validates a request without allocating resources.

## NPU count and device numbering

The live Slurm submission validator accepts **2, 4, 6, 8, 10, 12, 14, or 16 NPU units per node**; its message directs larger requests to multiple nodes. A one-unit request is rejected. The validator explicitly says:

> NPU 是一卡双芯，提交 2 张卡作业是按 1 张物理卡计费。

Thus `--gres=npu:2` means **two logical chips / one physical 910C card**, not two physical cards. The verified chips each expose about 64 GiB HBM. Do not infer prices, utilization warning thresholds, or quota rules from this count policy; confirm those separately.

Slurm device isolation and the Ascend runtime remap allocated chips to job-local logical indices. In the verified two-chip allocation, physical devices 8 and 9 were exposed as logical `npu:0` and `npu:1`. Passing physical IDs through `ASCEND_RT_VISIBLE_DEVICES=8,9` failed because the runtime accepted indices in `[0,2)`.

For this Slurm environment, leave `ASCEND_RT_VISIBLE_DEVICES` unset and select the job-local index in the application. Verify `torch_npu.npu.device_count()` inside the allocation. Do not equate host `npu-smi` IDs with application indices or copy this mapping assumption to an unverified container.

Requesting two chips does not make training parallel automatically. Use DDP for one distributed run, or separate processes explicitly assigned logical indices 0 and 1 for independent experiments. Choose according to the requested experiment. Do not assume a particular project's `--npu` flag exists in another program.

## Verified Python environment

The following environment ran BF16 PyTorch training and autoregressive evaluation in September 2026:

- Python: `/data/anaconda3/envs/pytorch-npu/bin/python` (3.11.14).
- PyTorch 2.5.1 and `torch_npu` 2.5.1.post3.
- CANN environment: `/usr/local/Ascend/ascend-toolkit/set_env.sh` (observed CANN 8.5.0).
- User home observed at `/data/user/<username>`; verify paths for the current account.

Use the existing compatible NPU environment as the base for a project virtualenv, inspecting installed versions before changing dependencies. A project may need `--system-site-packages` to access the installed torch/torch_npu stack. Avoid replacing the shared environment or installing CUDA wheels over it.

An illustrative two-chip Slurm script (replace the project path and application command; size resources for the actual task):

```bash
#!/bin/bash
#SBATCH --job-name=npu-train
#SBATCH --partition=a320m2tn910cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=npu:2
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
unset ASCEND_RT_VISIBLE_DEVICES
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd /absolute/project/path
# This command must implement the intended use of both allocated chips.
.venv/bin/python train.py <application arguments>
```

Create the log directory before submitting and set the submission working directory explicitly. All `#SBATCH` directives must precede executable commands. For CPU jobs, use a currently available CPU partition and omit the NPU GRES; inspect modules or the installed Python environment as needed.

## Monitor useful work

Use `squeue`, `sacct`, application progress, and errors together. Inspect `npu-smi info` inside the allocation for the allocated hardware. Distinguish AICore utilization, composite NPU utilization, and HBM occupancy; memory usage alone does not measure useful compute. Measure a time window and distinguish initialization, training, validation, and generation.

If utilization is low, measure input preparation, compute, and communication before changing resource counts. Choose parallelism and batch size for the actual workload; do not alter the experimental design merely to raise a utilization metric.

Do not present an invented utilization percentage as a school policy. Never put credentials in a skill, repository, or logs.
