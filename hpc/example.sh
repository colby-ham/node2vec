#!/usr/bin/bash

#SBATCH --job-name=bprmf
#SBATCH --output=logs/%x-%j.out
#SBATCH -A st
#SBATCH -p shared_dlt
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH -t 3-23:59:00

module purge
module load cuda/9.2.148 
module load python/anaconda3.2019.3
module load gcc/5.2.0
source /share/apps/python/anaconda3.2019.3/etc/profile.d/conda.sh
source activate node2vec

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISISBLE_DEVICES}"
dataset=example
printf "Running node2vec on dataset=${dataset}"

# Code location
REPO_DIR=~/recommendation/node2vec

# Shared memory location
SHARED_DIR=/projects/streaming_graph/recommendations
N2V_DIR=$SHARED_DIR/embeddings/node2vec
DATASET_DIR=$N2V_DIR/$dataset
OUTPUT_DIR=$DATASET_DIR/output

EMBEDDING_FILENAME=$OUTPUT_DIR/${dataset}.emb
EMBEDDING_MODEL_FILENAME=$OUTPUT_DIR/${dataset}.model


printf "Going to REPO_DIR=${REPO_DIR}"
cd $REPO_DIR
python example.py $EMBEDDING_FILENAME $EMBEDDING_MODEL_FILENAME
printf "Finished"
