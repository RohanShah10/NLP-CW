# Patronising and Condescending Language (PCL) Detection

This repository contains the code, trained model, and prediction outputs
for a **binary classification system that detects Patronising and
Condescending Language (PCL)**. The task is based on **SemEval-2022 Task
4: Patronising and Condescending Language Detection**.

## Overview

The project fine-tunes a **RoBERTa sequence classification model** for
detecting PCL in news text. A major challenge in the dataset is the
**significant class imbalance**, with only **approximately 9.5% positive
examples**.

To address this imbalance, training uses a **class-weighted
Cross-Entropy loss function** applied across the entire training set.
This allows the model to learn from all available data while assigning
greater importance to minority-class examples.

This approach performs better than the standard baseline used in the
task, which relies on **random undersampling and therefore discards a
large portion of the training data**.

## Repository Structure

    BestModel/
        ├── model files
        ├── training notebook (.ipynb)

    dev.txt
    test.txt
    README.md

**Descriptions**

-   **/BestModel/** Contains the final trained RoBERTa model files along
    with the main Jupyter Notebook used for preprocessing, training, and
    evaluation. Note: GitHub may not preview the notebook correctly ---
    download it locally to view.

-   **dev.txt** Binary predictions (`0` or `1`) corresponding
    line-by-line with the official **development set**.

-   **test.txt** Binary predictions (`0` or `1`) corresponding
    line-by-line with the official **test set**.

-   **README.md** Documentation and instructions for reproducing the
    model.

## Required Datasets

To reproduce this project, the official SemEval datasets must be placed
in the project directory alongside the notebook.

Required files:

    dontpatronizeme_pcl.tsv
    train_semeval_parids-labels.csv
    dev_semeval_parids-labels.csv
    task4_test.tsv

## Pre-trained Model Weights

Due to GitHub file size limits, the main model weights file
(`model.safetensors`) is hosted externally.

To run inference without retraining the model:

1.  Download the weights from the link below.
2.  Create a directory named:

```{=html}
<!-- -->
```
    /Model/

3.  Place the following inside `/Model/`:
    -   `model.safetensors`
    -   All files from `/BestModel/` **except the `.ipynb` notebook**
4.  Ensure the notebook remains outside this directory before running
    the evaluation cells.

Download the weights here:

    model.safetensors

## Reproducing the Model

### 1. Environment Setup

Ensure **Python 3.8 or newer** is installed.

Install the required dependencies:

``` bash
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn
```

### 2. File Placement

Place the dataset files listed above in the **same working directory as
the Jupyter Notebook** located inside `/BestModel/`.

### 3. Execution

Open the notebook and run the cells sequentially. The notebook is
organised into the following stages:

**Data Preprocessing**

-   Loads the datasets
-   Aligns examples with the official dev and test IDs
-   Constructs the required DataFrames

**Tokenization**

-   Uses the RoBERTa tokenizer
-   Inputs are truncated to `max_length = 128` to reduce VRAM usage and
    allow larger batch sizes during training

**Training**

-   Initializes the HuggingFace `Trainer`
-   Uses a **custom class-weighted loss function** to address dataset
    imbalance

*(Skip this stage if using the provided pretrained weights.)*

**Evaluation**

-   Generates prediction files (`dev.txt` and `test.txt`)
-   Outputs evaluation metrics including **Precision, Recall, and F1
    score**
