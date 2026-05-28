#!/bin/bash
#SBATCH --job-name=bf_bench
#SBATCH --account=def-aevans
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=2:00:00
#SBATCH --output=logs/bf_bench_%j.out
#SBATCH --error=logs/bf_bench_%j.err

# Bench runner: persistence + moving_average + ar.
# CPU-only — no --gres=gpu line. Single process, all folds sequentially
# (these predictors are cheap; fold parallelism not worth the array overhead).
#
# To diagnose:
#   grep EXPERIMENT_CONFIG logs/bf_bench_${SLURM_JOB_ID}.out
#   grep HARDWARE          logs/bf_bench_${SLURM_JOB_ID}.out

set -euo pipefail
mkdir -p logs

module load apptainer

# ---- PATHS --------------------------------------------------------------
PROJECT_DIR=/home/tamires/projects/rpp-aevans-ab/tamires/BrainForecast
BASE=${PROJECT_DIR}/inputs
BRAIN_PATH=${BASE}/brain_observed_dynamic_per_second.parquet
STIMULUS_PATH=${BASE}/known_stimuli_per_second.parquet
STATIC_PATH=${BASE}/static_participant_features.csv
SIF=/home/tamires/projects/rpp-aevans-ab/tamires/singularity/cinematic_forecast_cuda.sif

# ---- EXPERIMENT KNOBS ---------------------------------------------------
# Predictor selection. Comma-separated. Bundle aliases: bench, tft, all.
PREDICTORS=bench

# Data / CV
HORIZON_MIN=1
N_TARGETS=1
KNOWN_DYNAMIC_CAP=100

# Predictor-specific kwargs (each takes effect only if that predictor is in
# PREDICTORS — listed here so all knobs are visible in one place)
AR_P=5
MA_K=5

# Output verbosity: 0 = basic (scores.csv + resource_profile.csv only),
#                   1 = full (also plots: horizon_curves, per_cohort, heatmap)
FULL_OUTPUTS=0

# Where to write. No fold-parallelism here, so a single output dir.
OUTPUT_DIR=${PROJECT_DIR}/outputs/bench_h${HORIZON_MIN}_n${N_TARGETS}
# -------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"

echo "=== Bench run (CPU-only, sequential folds) ==="
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Predictors: ${PREDICTORS}"
echo "Horizon   : ${HORIZON_MIN} min"
echo "N targets : ${N_TARGETS}, Known-dyn cap: ${KNOWN_DYNAMIC_CAP}"
echo "AR/MA     : ar_p=${AR_P} ma_k=${MA_K}"
echo "Output    : ${OUTPUT_DIR}"

# --full-outputs is a boolean flag; convert FULL_OUTPUTS=1 to the flag, else omit.
FULL_OUTPUTS_FLAG=""
if [ "${FULL_OUTPUTS}" -eq 1 ]; then
  FULL_OUTPUTS_FLAG="--full-outputs"
fi

# CPU-only: no --nv flag, no fold-parallel flags.
apptainer exec \
  --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
  --bind "${BASE}:${BASE}" \
  --bind "/home/tamires/scratch:/home/tamires/scratch" \
  "${SIF}" \
  bash -lc "
    cd '${PROJECT_DIR}' &&
    export OMP_NUM_THREADS=4 &&
    export MKL_NUM_THREADS=4 &&
    export TOKENIZERS_PARALLELISM=false &&
    python3 -m brain_forecast run \
      --predictors '${PREDICTORS}' \
      --brain '${BRAIN_PATH}' \
      --stimulus '${STIMULUS_PATH}' \
      --static '${STATIC_PATH}' \
      --output-dir '${OUTPUT_DIR}' \
      --horizon-min ${HORIZON_MIN} \
      --n-targets ${N_TARGETS} \
      --known-dynamic-cap ${KNOWN_DYNAMIC_CAP} \
      --ar-p ${AR_P} \
      --ma-k ${MA_K} \
      ${FULL_OUTPUTS_FLAG}
  "

echo "bench run finished. Outputs in ${OUTPUT_DIR}"

# ── SLURM resource accounting ─────────────────────────────────────────────
sleep 30
USAGE_FILE="${OUTPUT_DIR}/slurm_usage.txt"
{
  echo "=== job ${SLURM_JOB_ID} (${SLURM_JOB_NAME}) ==="
  echo "--- seff ---"
  seff "${SLURM_JOB_ID}" 2>/dev/null || echo "seff unavailable"
  echo
  echo "--- sacct ---"
  sacct -j "${SLURM_JOB_ID}" --units=G \
    --format=JobID,JobName,Elapsed,Timelimit,ReqMem,MaxRSS,MaxVMSize,AllocCPUS,TotalCPU,State \
    2>/dev/null || echo "sacct unavailable"
} | tee "${USAGE_FILE}"
echo "Wrote SLURM accounting to ${USAGE_FILE}"
