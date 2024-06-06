# Enhanced Question-Answering System

## Overview
This project enhances a question-answering system using attention-based models like BERT.

## Setup
1. Clone the repository.
2. Create a virtual environment and install dependencies:
    ```bash
    conda create --name qa-system python=3.8
    conda activate qa-system
    pip install -r requirements.txt
    ```

## Training
1. Prepare the dataset and fine-tune the model:
    ```bash
    python train.py
    ```

## Evaluation
1. Evaluate the model:
    ```bash
    python evaluate.py
    ```