import logging
import pandas as pd
from src.preprocessing.read_data import csv_reader
from src.preprocessing.preprocessing import apply_preprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__=='__main__':
  df = csv_reader('data/raw/imdb_dataset.csv')
  
  # Pre-Processing
  print('Start preprocessing')
  try:
    df_preprocessed = apply_preprocessing(df)
    logger.info(f'Finished preprocessing. Results: {df_preprocessed.info()}')
  except Exception as e:
    logger.error(f'Error while preprocessing: {e}')
    exit(1)
  
  # Word Vectorization
  
  
  # Model Training
  
  
  # Model Testing
  
  pass