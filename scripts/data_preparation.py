import pandas as pd
from datasets import load_dataset

def load_and_preprocess_data():
    dataset = load_dataset('squad')
    train_data = dataset['train']
    validation_data = dataset['validation']

    # Example preprocessing steps can go here

    train_data.to_csv('data/train.csv', index=False)
    validation_data.to_csv('data/validation.csv', index=False)

if __name__ == "__main__":
    load_and_preprocess_data()
