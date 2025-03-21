# Forensic Iris Image-Based Post-Mortem Interval Estimation 

Official repository for the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2025) paper: **IEEEXplore | [ArXiv](https://arxiv.org/pdf/2404.10172)**.

### Abstract
Post-mortem iris recognition is an emerging application of iris-based human identification in a forensic setup. One factor that may be useful in conditioning iris recognition methods is the tissue decomposition level, which is correlated with the post-mortem interval (PMI), i.e., the number of hours that have elapsed since death. PMI, however, is not always available, and its precise estimation remains one of the core challenges in forensic examination. This paper presents the first known to us method of the PMI estimation directly from iris images captured after death. To assess the feasibility of the iris-based PMI estimation, we designed models predicting the PMI from (a) near-infrared (NIR), (b) visible (RGB), and (c) multispectral (RGB+NIR) forensic iris images. Models were evaluated following a 10-fold cross-validation, in (S1) sample-disjoint, (S2) subject-disjoint, and (S3) cross-dataset scenarios. We explore two data balancing techniques for S3: resampling-based balancing (S3-real), and synthetic data-supplemented balancing (S3-synthetic). We found that using the multispectral data offers a spectacularly low mean absolute error (MAE) of ≈3.5 hours in the scenario (S1), a bit worse MAE ≈17.5 hours in the scenario (S2), and MAE ≈45.77 hours in the scenario (S3). Additionally, supplementing the training set with synthetically-generated forensic iris images (S3-synthetic) significantly enhances the models' ability to generalize to new NIR, RGB and multispectral data collected in a different lab. This suggests that if the environmental conditions are favorable (e.g., bodies are kept in low temperatures), forensic iris images provide features that are indicative of the PMI and can be automatically estimated.

## Running the models

### Create Environment
Set up Python environment using Conda:

``
conda env create -f environment.yml
``

### Prediction and Training
1.) Run the following bash script to train the subject disjoint models:

``
./train_sub_disj.sh
``

2.) Run the following bash script to train the cross-dataset models:

``
./train_cross_dt.sh
``

***Note: Make sure you have provided all the valid paths in the script.***

### Obtaining the Trained Model Weights
The trained model weights can be downloaded from this [Box folder](https://notredame.box.com/s/5nugb0k602d28va3exlkzkec2wzokyh3).

## Citation
Please cite our paper if you use any part of our code or data.

```
@InProceedings{Bhuiyan_2025_WACV,
    author    = {Bhuiyan, Rasel Ahmed and Czajka, Adam},
    title     = {Forensic Iris Image-Based Post-Mortem Interval Estimation},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {4258-4267}
}
```
