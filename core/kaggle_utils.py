import os
import json
import kagglehub
import pandas as pd
from pathlib import Path

def download_dataset(dataset_name, path='data'):
    """
    Download a dataset from Kaggle using kagglehub.
    
    Args:
        dataset_name (str): Name of the dataset in format 'username/dataset-name'
        path (str): Path to save the dataset
    
    Returns:
        str: Path to the downloaded dataset
    """
    # Create data directory if it doesn't exist
    data_dir = Path(path)
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Download the dataset using kagglehub
        dataset_path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset downloaded to: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        raise

def load_dataset(file_path):
    """
    Load a dataset from a file.
    
    Args:
        file_path (str): Path to the dataset file
    
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel files.") 