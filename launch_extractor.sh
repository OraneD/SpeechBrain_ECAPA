#!/bin/bash
#SBATCH --job-name=ExtractEcapa     # Nom du job
#SBATCH --ntasks=1                       # Nombre de tâches
#SBATCH --gres=gpu:4                      # Nombre de GPU
#SBATCH --cpus-per-task=40               # Nombre de CPU par tâche
#SBATCH --qos=qos_gpu-t3     # Quality of Service qos_gpu-dev / qos_gpu-t3
#SBATCH --partition=gpu_p13               # Partition GPU
#SBATCH -A yjs@v100                       # Allocation de ressources (compte utilisateur)
#SBATCH -C v100-32g                       # Type de GPU (V100 avec 32Go)
#SBATCH --time=20:00:00                   # Temps limite (1h)
#SBATCH --output=ExtractEcapa_%j.out         # Fichier de sortie
#SBATCH --error=ExtractEcapa_%j.err


source ../miniconda3/bin/activate ~/.conda/envs/anon_B3
export PYTHONPATH=$PWD

WAVSCP="$1"
CKPT="$2"
OUT=$3

python speaker_embeddings/extractor.py --wav_scp "${WAVSCP}" --checkpoint "${CKPT}" --output "${OUT}"
