import logging
import pandas as pd
from src.preprocessing.read_data import csv_reader
from src.preprocessing.preprocessing import apply_preprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__=='__main__':
  df = csv_reader('data/raw/imdb_dataset.csv')
  
  # Pre-Processing
  apply_preprocessing(df)
  df = csv_reader('data/preprocessed/preprocessed_dataset.csv')
  
  # Word Vectorization
  
  
  
  # Model Training
  
  
  # Model Testing
  
  pass