#!/bin/bash
#$ -M rbhuiyan@nd.edu
#$ -m e
#$ -q gpu
#$ -l gpu_card=4
#$ -l h=qa-a10*
#$ -N syn-m-tn

# Load module
source activate pmiestimation

# Change directory to where the Python script is located
cd ~/PMI-Estimation-main

# Change this for different data combination
data_type="multi"     # NIR, RGB, multi
trainset="warsaw"     # nij, warsaw
checkpoint_base="./models-checkpoint/ViT/syn-disj/"

# Define model architectures
models=("vgg" "resnet" "inception" "densenet" "vit" "domainresnet")

# Model parameter
batch_size=128
solver_name="Adam"
lr=0.0001
num_epochs=300


# Image directories
nir_image_root_dir="./iris-recognition-dataset/warsaw-nij-cropped-nir-images/"
rgb_image_root_dir="./iris-recognition-dataset/warsaw-nij-cropped-rgb-images/"

# Conditional selection of train and test data based on the trainset variable
if [[ $trainset == "warsaw" ]]; then
    nir_train_data="./train-testset/bal-ds-disj-metadata/bal-warsaw-NIR-metadata.txt"
    nir_test_data="./train-testset/ds-disj-metadata/nij-NIR-metadata.txt"

    rgb_train_data="./train-testset/bal-ds-disj-metadata/bal-warsaw-RGB-metadata.txt"
    rgb_test_data="./train-testset/ds-disj-metadata/nij-RGB-metadata.txt"

    multi_train_data="./train-testset/bal-ds-disj-metadata/bal-warsaw-multispectral-metadata.txt"
    multi_test_data="./train-testset/ds-disj-metadata/nij-multispectral-metadata.txt"

    if [[ $data_type == "multi" ]]; then
        nir_syn_image_root_dir="./iris-recognition-dataset/syn-nij-NIR-images/"
        rgb_syn_image_root_dir="./iris-recognition-dataset/syn-nij-RGB-images/"
    else
        nir_syn_image_root_dir="./iris-recognition-dataset/syn-warsaw-NIR-images/"
        rgb_syn_image_root_dir="./iris-recognition-dataset/syn-warsaw-RGB-images/"
    fi

    nir_synthetic_data="./train-testset/bal-syn-metadata/bal-syn-warsaw-NIR-metadata.txt"
    rgb_synthetic_data="./train-testset/bal-syn-metadata/bal-syn-warsaw-RGB-metadata.txt"
    multi_synthetic_data="./train-testset/bal-syn-metadata/bal-syn-warsaw-multispectral-metadata.txt"

    checkpoint="${checkpoint_base}testset-nij/"
elif [[ $trainset == "nij" ]]; then
    nir_train_data="./train-testset/bal-ds-disj-metadata/bal-nij-NIR-metadata.txt"
    nir_test_data="./train-testset/ds-disj-metadata/warsaw-NIR-metadata.txt"

    rgb_train_data="./train-testset/bal-ds-disj-metadata/bal-nij-RGB-metadata.txt"
    rgb_test_data="./train-testset/ds-disj-metadata/warsaw-RGB-metadata.txt"

    multi_train_data="./train-testset/bal-ds-disj-metadata/bal-nij-multispectral-metadata.txt"
    multi_test_data="./train-testset/ds-disj-metadata/warsaw-multispectral-metadata.txt"

    nir_syn_image_root_dir="./iris-recognition-dataset/syn-nij-NIR-images/"
    rgb_syn_image_root_dir="./iris-recognition-dataset/syn-nij-RGB-images/"

    nir_synthetic_data="./train-testset/bal-syn-metadata/bal-syn-nij-NIR-metadata.txt"
    rgb_synthetic_data="./train-testset/bal-syn-metadata/bal-syn-nij-RGB-metadata.txt"
    multi_synthetic_data="./train-testset/bal-syn-metadata/bal-syn-nij-multispectral-metadata.txt"

    checkpoint="${checkpoint_base}testset-warsaw/"
else
    echo "Error: Invalid trainset value. Use 'warsaw' or 'nij'."
    exit 1
fi

# Loop through each architecture
for arch in "${models[@]}"; do
    echo "Running model with $data_type data and $arch architecture [Trainset: $trainset]"

    # Call the Python script with the parsed arguments
    python3 train_cross_dataset_model.py \
        --nir_image_root_dir $nir_image_root_dir \
        --rgb_image_root_dir $rgb_image_root_dir \
        --nir_syn_image_root_dir $nir_syn_image_root_dir \
        --rgb_syn_image_root_dir $rgb_syn_image_root_dir \
        --nir_train_data $nir_train_data \
        --nir_test_data $nir_test_data \
        --rgb_train_data $rgb_train_data \
        --rgb_test_data $rgb_test_data \
        --multi_train_data $multi_train_data \
        --multi_test_data $multi_test_data \
        --nir_synthetic_data $nir_synthetic_data \
        --rgb_synthetic_data $rgb_synthetic_data \
        --multi_synthetic_data $multi_synthetic_data \
        --data_type $data_type \
        --arch $arch \
        --solver_name $solver_name \
        --batch_size $batch_size \
        --lr $lr \
        --num_epochs $num_epochs \
        --checkpoint $checkpoint \
        --weight_decay \
        --merge_syn 
        # --pretrained
done
