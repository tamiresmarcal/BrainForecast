#!/bin/bash
#SBATCH --job-name=bf_tft_complete
#SBATCH --account=def-aevans
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=90G
#SBATCH --time=3-00:00:00
#SBATCH --array=0
#SBATCH --output=logs/bf_tft_fold%a_%A.out
#SBATCH --error=logs/bf_tft_fold%a_%A.err

# TFT runner with fold parallelism AND channel projection.
# Each array task runs ONE fold on its own H100. Aggregate after with:
#   python -m brain_forecast aggregate --output-dir ${OUTPUT_DIR}
#
# What's different from sbatch_tft_complete.sh:
#   * --tft-channel-proj enabled (per-channel learned projection)
#   * --tft-channel-proj-dim 64 (target dim per channel; channels with
#     raw_dim <= 64 stay at their raw_dim)
#   * Per-channel slot count after projection is ~12*64 + 48 surprise
#     scalars + the few 1-d channels (mov_onset etc) -> O(800), down
#     from the raw 4900.
#
# To diagnose:
#   grep EXPERIMENT_CONFIG logs/bf_tft_fold*_${SLURM_ARRAY_JOB_ID}.out
#   grep CHANNEL_PROJ      logs/bf_tft_fold*_${SLURM_ARRAY_JOB_ID}.out
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
KNOWN_DYNAMIC_CAP=4947    # full feature set; channel projection will compress it
N_FOLDS=1                 # must match --array size above

# TFT hyperparameters (used only when 'tft' is in PREDICTORS)
MAX_EPOCHS=15
HIDDEN_SIZE=64
ATTENTION_HEAD_SIZE=4
DROPOUT=0.1
BATCH_SIZE=256
LEARNING_RATE=2.5e-4
NUM_WORKERS=12

# Channel projection (Algonauts-style, see brain_forecast/predictors/channel_projection.py)
# Pass --tft-channel-proj to enable; channels with raw_dim <= TFT_CHANNEL_PROJ_DIM
# stay at their raw_dim (identity, no params). Surprise/uncertainty columns and
# any 1-d channels pass through unchanged.
TFT_CHANNEL_PROJ=1                  # 1 = on, 0 = off (legacy path)
TFT_CHANNEL_PROJ_DIM=64
TFT_CHANNEL_PATTERN='^mov_(?P<channel>.+)_(?P<dim>\d+)$'

# Output verbosity: 0 = basic (scores.csv + resource_profile.csv only),
#                   1 = full (also plots: horizon_curves, per_cohort, heatmap)
FULL_OUTPUTS=0

# Where to write. Pattern encodes the most-varied knobs.
PROJ_TAG=""
if [ "${TFT_CHANNEL_PROJ}" -eq 1 ]; then
  PROJ_TAG="_proj${TFT_CHANNEL_PROJ_DIM}"
fi
OUTPUT_DIR=${PROJECT_DIR}/outputs/tft_w${WINDOW_MIN}_h${HORIZON_MIN}_s${STRIDE}_n${N_TARGETS}${PROJ_TAG}
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
echo "Channel proj: ${TFT_CHANNEL_PROJ} (target dim ${TFT_CHANNEL_PROJ_DIM})"
echo "Output    : ${FOLD_DIR}"

# Boolean flags: convert FULL_OUTPUTS=1 to the flag, else omit.
FULL_OUTPUTS_FLAG=""
if [ "${FULL_OUTPUTS}" -eq 1 ]; then
  FULL_OUTPUTS_FLAG="--full-outputs"
fi
# Channel projection flag
CHANNEL_PROJ_FLAG=""
if [ "${TFT_CHANNEL_PROJ}" -eq 1 ]; then
  CHANNEL_PROJ_FLAG="--tft-channel-proj"
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
      ${CHANNEL_PROJ_FLAG} \
      --tft-channel-proj-dim ${TFT_CHANNEL_PROJ_DIM} \
      --tft-channel-pattern '${TFT_CHANNEL_PATTERN}' \
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
