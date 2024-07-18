# DCCNV: Denoising and Circular Binary Segmentation for CNV Detection

## Overview
This repository contains the implementation of the DCCNV pipeline for copy number variation (CNV) detection in single-cell DNA sequencing data. The pipeline includes data preprocessing, graph analysis, denoising using a diffusion process, contrastive learning, and segmentation using Circular Binary Segmentation (CBS).

## Project Structure
DCCNV/
├── data/
│ └── A4_readcounts_processed.tsv
├── src/
│ ├── data_preprocessing.py
│ ├── graph_analysis.py
│ ├── denoising.py
│ ├── contrastive_learning.py
│ ├── segmentation.py
│ └── main.py
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore


## Requirements
The required Python packages can be installed using:
pip install -r requirements.txt



## Usage
To run the DCCNV pipeline, use the following command:

python src/main.py data/data.tsv


## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Data Preprocessing
The `data_preprocessing.py` module handles data loading and normalization. It includes the `DataPreprocessor` class with methods to load and normalize the data.

## Graph Analysis
The `graph_analysis.py` module performs graph analysis including k-NN graph construction and affinity matrix computation. It includes the `GraphAnalyzer` class with methods to compute adaptive k, construct the k-NN graph, and compute the affinity matrix.

## Denoising
The `denoising.py` module implements the denoising technique using a multi-scale diffusion process. It includes the `Denoiser` class with methods to apply the diffusion process and denoise the data.

## Contrastive Learning
The `contrastive_learning.py` module defines and trains a contrastive learning model using a Siamese network architecture. It includes the `ContrastiveLearningModel` class with methods to create the base network, create the contrastive model, and compile and train the model.

## Segmentation
The `segmentation.py` module performs segmentation using Circular Binary Segmentation (CBS). It includes the `Segmenter` class with methods to segment the input signal, convert the segmented result to an array, normalize the segmented result, and plot the segmented signal against the ground truth.

## Main Script
The `main.py` script runs the entire DCCNV pipeline. It includes data preprocessing, graph analysis, denoising, contrastive learning, and segmentation steps. The script also plots the segmented signal against the ground truth for comparison.

### Example Usage
To run the pipeline with the provided example data, use the following command:
python src/main.py data/data.tsv

python src/main.py data/A4_readcounts_processed.tsv
