#!/bin/bash
#SBATCH --job-name=TrainECAPA     # Nom du job
#SBATCH --ntasks=1                       # Nombre de tâches
#SBATCH --gres=gpu:4                      # Nombre de GPU
#SBATCH --cpus-per-task=40               # Nombre de CPU par tâche
#SBATCH --qos=qos_gpu-t3     # Quality of Service qos_gpu-dev / qos_gpu-t3
#SBATCH --partition=gpu_p13               # Partition GPU
#SBATCH -A yjs@v100                       # Allocation de ressources (compte utilisateur)
#SBATCH -C v100-32g                       # Type de GPU (V100 avec 32Go)
#SBATCH --time=20:00:00                   # Temps limite (1h)
#SBATCH --output=TrainECAPA_%j.out         # Fichier de sortie
#SBATCH --error=TrainECAPA_%j.err

if [ -z "$1" ]; then
  echo "Usage: sbatch train_ecapa.sh _anon_B5"
  exit 1
fi

ANON_SUFFIX="$1"

source ../miniconda3/bin/activate ~/.conda/envs/anon_B3
export PYTHONPATH=$PWD


python run_training.py \
--config exp_config.yaml \
--gpu_ids 0,1,2,3 \
--overwrite "{
\"anon_data_suffix\": \"${ANON_SUFFIX}\"
}"
