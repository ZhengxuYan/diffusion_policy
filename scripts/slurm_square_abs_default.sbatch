#!/bin/bash
#SBATCH --job-name=dp-square-abs
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodelist=iris6
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/iris/u/jasonyan/logs/%x-%j.out
#SBATCH --error=/iris/u/jasonyan/logs/%x-%j.err

set -euo pipefail

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate robodiff

cd /iris/u/jasonyan/repos/diffusion_policy
mkdir -p /iris/u/jasonyan/logs

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0

TASK="${1:-square_image_abs}"

if [[ "$TASK" != "square_image_abs" && "$TASK" != "square_image_abs_wrist_only" ]]; then
  echo "Unsupported task: $TASK"
  echo "Use square_image_abs or square_image_abs_wrist_only"
  exit 2
fi

python train.py \
  --config-name=train_diffusion_unet_image_workspace \
  task="$TASK" \
  training.num_epochs=51 \
  logging.mode=disabled \
  training.device=cuda:0
