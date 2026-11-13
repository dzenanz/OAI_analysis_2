#!/bin/bash
# A test run of oai-analysis on POCUS: the entire machine (4 GPUs, 12/24 CPU threads).
# Submit with `sbatch pocus_test.sbatch`; follow with `tail -f oai-test-JOBID.out`.
#SBATCH --job-name=oai-test
#SBATCH --nodelist=pocus
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=oai-test-%j.out

set -euo pipefail

# --- edit these ---
ENV=/data/users/dzenan.zukic/oai/.venv            # the virtual environment with oai-analysis installed
PROJECT=/data/users/dzenan.zukic/oai/pocus_test   # created on the first run, resumed on the next ones
MANIFEST=/data/users/dzenan.zukic/oai/OAI_analysis_2/oai-analysis-test-data/study_uid_to_vol_path.csv
DATA_ROOT=/home/local/KHQ/dzenan.zukic/mnt/OAI    # the folder the manifest's paths are relative to
PATIENTS=9000099,9000296,9000622,9001104,9001897,9002316,9002817,9003380
# ------------------

source "$ENV/bin/activate"

echo "Job $SLURM_JOB_ID on $(hostname), $(date)"
echo "CPUs this job may use: $(nproc); GPUs: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L
python -c "import torch, itk, oai_analysis; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), \
torch.cuda.device_count(), 'GPUs; ITK', itk.Version.GetITKVersion())"

if [ ! -f "$PROJECT/project.yaml" ]; then
    oai-analysis init "$PROJECT" --manifest "$MANIFEST" --data-root "$DATA_ROOT" --patients "$PATIENTS"
fi

# GPU use every 5 s, to see whether the GPUs stay busy
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 5 \
    > "$PROJECT/gpu_usage_$SLURM_JOB_ID.csv" &
MONITOR=$!
trap 'kill $MONITOR 2>/dev/null || true' EXIT

start=$(date +%s)
oai-analysis run "$PROJECT" || echo "Some cases failed (see above)."
seconds=$(( $(date +%s) - start ))
cases=$(( $(wc -l < "$PROJECT/cases.csv") - 1 ))
echo "Wall time ${seconds} s for ${cases} cases: $(( seconds / cases )) s per case"

oai-analysis status "$PROJECT" | head -3
