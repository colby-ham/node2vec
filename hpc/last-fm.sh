#!/bin/bash

#SBATCH --job-name=n2v_last-fm
#SBATCH --output=logs/%x-%j.out
#SBATCH -A st_graphs
#SBATCH -p shared_dlt
#SBATCH -n 1
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
dataset=last-fm
printf "Running node2vec on dataset=${dataset}\n"
date

# Code location
REPO_DIR=~/recommendation/node2vec

# Shared memory location
SHARED_DIR=/projects/streaming_graph/recommendations
N2V_DIR=$SHARED_DIR/embeddings/node2vec
DATASET_DIR=$N2V_DIR/$dataset
INPUT_DIR=$DATASET_DIR/input
OUTPUT_DIR=$DATASET_DIR/output

EDGELIST_FILENAME=$INPUT_DIR/${dataset}.edgelist
EMBEDDING_FILENAME=$OUTPUT_DIR/${dataset}.emb
EMBEDDING_MODEL_FILENAME=$OUTPUT_DIR/${dataset}.model


printf "Going to REPO_DIR=${REPO_DIR}\n"
cd $REPO_DIR
python example.py $EDGELIST_FILENAME $EMBEDDING_FILENAME $EMBEDDING_MODEL_FILENAME
printf "Finished"
date
