#!/bin/bash
#SBATCH --job-name=tft_bigger_layers
#SBATCH --account=def-aevans
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=15G
#SBATCH --time=8:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/bf_tft_fold%a_%A.out
#SBATCH --error=logs/bf_tft_fold%a_%A.err

# TFT runner with fold parallelism.
# Each array task runs ONE fold on its own H100. Aggregate after with:
#   python -m brain_forecast aggregate --output-dir ${OUTPUT_DIR}
#
# To diagnose:
#   grep EXPERIMENT_CONFIG logs/bf_tft_fold*_${SLURM_ARRAY_JOB_ID}.out
#   grep HARDWARE          logs/bf_tft_fold*_${SLURM_ARRAY_JOB_ID}.out
#   grep "bf.train"        logs/bf_tft_fold*_${SLURM_ARRAY_JOB_ID}.out

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
# Predictor selection. Comma-separated list. Bundle aliases: bench, tft, all.
PREDICTORS=tft

# Data / CV
WINDOW_MIN=1
HORIZON_MIN=1
STRIDE=1
N_TARGETS=100
KNOWN_DYNAMIC_CAP=100
N_FOLDS=2                # must match --array size above

# TFT hyperparameters (used only when 'tft' is in PREDICTORS)
MAX_EPOCHS=10
HIDDEN_SIZE=64
ATTENTION_HEAD_SIZE=4
DROPOUT=0.1
BATCH_SIZE=1024
LEARNING_RATE=1e-3
NUM_WORKERS=12


# Output verbosity: 0 = basic (scores.csv + resource_profile.csv only),
#                   1 = full (also plots: horizon_curves, per_cohort, heatmap)
FULL_OUTPUTS=0

# Where to write. Pattern encodes the most-varied knobs.
OUTPUT_DIR=${PROJECT_DIR}/outputs/tft_w${WINDOW_MIN}_h${HORIZON_MIN}_s${STRIDE}_n${N_TARGETS}
# -------------------------------------------------------------------------

FOLD_DIR="${OUTPUT_DIR}/fold_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${FOLD_DIR}"

echo "=== Array task ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT} ==="
echo "Job ID    : ${SLURM_JOB_ID} (array job ${SLURM_ARRAY_JOB_ID})"
echo "Fold      : ${SLURM_ARRAY_TASK_ID} of ${N_FOLDS}"
echo "Predictors: ${PREDICTORS}"
echo "Window    : ${WINDOW_MIN} min, Horizon: ${HORIZON_MIN} min, Stride: ${STRIDE}"
echo "N targets : ${N_TARGETS}, Known-dyn cap: ${KNOWN_DYNAMIC_CAP}"
echo "TFT hyper : hidden=${HIDDEN_SIZE} heads=${ATTENTION_HEAD_SIZE} bs=${BATCH_SIZE} lr=${LEARNING_RATE} epochs=${MAX_EPOCHS} workers=${NUM_WORKERS}"
echo "Output    : ${FOLD_DIR}"

# --full-outputs is a boolean flag; convert FULL_OUTPUTS=1 to the flag, else omit.
FULL_OUTPUTS_FLAG=""
if [ "${FULL_OUTPUTS}" -eq 1 ]; then
  FULL_OUTPUTS_FLAG="--full-outputs"
fi

apptainer exec --nv \
  --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
  --bind "${BASE}:${BASE}" \
  --bind "/home/tamires/scratch:/home/tamires/scratch" \
  "${SIF}" \
  bash -lc "
    cd '${PROJECT_DIR}' &&
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True &&
    export OMP_NUM_THREADS=4 &&
    export MKL_NUM_THREADS=4 &&
    export TOKENIZERS_PARALLELISM=false &&
    python3 -m brain_forecast run \
      --predictors '${PREDICTORS}' \
      --brain '${BRAIN_PATH}' \
      --stimulus '${STIMULUS_PATH}' \
      --static '${STATIC_PATH}' \
      --output-dir '${FOLD_DIR}' \
      --window-min ${WINDOW_MIN} \
      --horizon-min ${HORIZON_MIN} \
      --stride ${STRIDE} \
      --n-targets ${N_TARGETS} \
      --known-dynamic-cap ${KNOWN_DYNAMIC_CAP} \
      --fold-idx ${SLURM_ARRAY_TASK_ID} \
      --n-folds ${N_FOLDS} \
      --max-epochs ${MAX_EPOCHS} \
      --hidden-size ${HIDDEN_SIZE} \
      --attention-head-size ${ATTENTION_HEAD_SIZE} \
      --dropout ${DROPOUT} \
      --batch-size ${BATCH_SIZE} \
      --learning-rate ${LEARNING_RATE} \
      --num-workers ${NUM_WORKERS} \
      ${FULL_OUTPUTS_FLAG}
  "

echo "Fold ${SLURM_ARRAY_TASK_ID} finished. Outputs in ${FOLD_DIR}."

# ── SLURM resource accounting ─────────────────────────────────────────────
sleep 30
USAGE_FILE="${FOLD_DIR}/slurm_usage.txt"
{
  echo "=== array job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} (fold ${SLURM_ARRAY_TASK_ID}) ==="
  echo "Full job id: ${SLURM_JOB_ID}"
  echo "--- seff ---"
  seff "${SLURM_JOB_ID}" 2>/dev/null || echo "seff unavailable"
  echo
  echo "--- sacct ---"
  sacct -j "${SLURM_JOB_ID}" --units=G \
    --format=JobID,JobName,Elapsed,Timelimit,ReqMem,MaxRSS,MaxVMSize,AllocCPUS,TotalCPU,State \
    2>/dev/null || echo "sacct unavailable"
} | tee "${USAGE_FILE}"
echo "Wrote SLURM accounting to ${USAGE_FILE}"
