#!/bin/bash
#$ -M rbhuiyan@nd.edu
#$ -m e
#$ -q gpu
#$ -l gpu_card=4
#$ -l h=qa-a10*
#$ -N sub-multi

# Load conda module
module load python

# Change directory to where the Python script is located
cd ~/PMI-Estimation-main

# Define data type and checkpoint directory
data_type="multi"
checkpoint="./models-checkpoint/sub-disj/$data_type/"

# Define model architectures to loop through
models=("vgg" "resnet" "inception" "densenet")

# Define model parameters
batch_size=128
solver_name="Adam"
lr=0.0001
num_epochs=1

# Define image directories and metadata path
nir_image_root_dir="./iris-recognition-dataset/warsaw-nij-cropped-nir-images/"
rgb_image_root_dir="./iris-recognition-dataset/warsaw-nij-cropped-rgb-images/"
metadata_file_path="./train-testset/sub-disj/multispectral/"

# Loop through each architecture
for arch in "${models[@]}"; do
    echo "Running model with $data_type data and $arch architecture"

    # Call the Python script with the parsed arguments
    python3 train_subject_disjoint_model.py \
        --nir_image_root_dir $nir_image_root_dir \
        --rgb_image_root_dir $rgb_image_root_dir \
        --metadata_file_path $metadata_file_path \
        --data_type $data_type \
        --arch $arch \
        --solver_name $solver_name \
        --batch_size $batch_size \
        --lr $lr \
        --num_epochs $num_epochs \
        --checkpoint $checkpoint \
        --weight_decay
        # --pretrained
done
