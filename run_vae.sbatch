#!/bin/bash
#SBATCH -J vae
#SBATCH --qos=comp3710
#SBATCH --account=comp3710
#SBATCH -p comp3710
#SBATCH --gres=gpu:1
#SBATCH -t 00:20:00
#SBATCH -c 2
#SBATCH -o logs/vae-%j.out
#SBATCH -e logs/vae-%j.err

set -euo pipefail

# 临时与依赖装到 /scratch，避免 home 空间问题
mkdir -p /scratch/$USER/tmp /scratch/$USER/python_libs
export TMPDIR=/scratch/$USER/tmp
export PIP_CACHE_DIR=/scratch/$USER/tmp/pipcache
export PYTHONPATH=/scratch/$USER/python_libs:$PYTHONPATH

cd ~/dawnbench
mkdir -p logs

# 轻量依赖(如需)放到 /scratch（多数节点已自带 torch/torchvision，可先不装）
python3 -m pip install --no-cache-dir --target=/scratch/$USER/python_libs numpy pillow matplotlib scikit-learn || true

# 运行 VAE（脚本里会从 /home/groups/comp3710/OASIS_preprocessed 读取数据）
python3 part4-vae.py
