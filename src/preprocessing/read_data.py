import ast
import logging
import os
import pandas as pd
import pickle
import threading

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def csv_reader(path: str =None)  -> pd.DataFrame:
  """
  Function to read a dataset stored in CSV format.
  
  Args:
    path: str = None -> Path to the dataset stored as CSV.
  
  Return:
    df: pd.DataFrame -> Returns the data as a dataframe.
  """
  try:  
    df = pd.read_csv(path)
    logger.info(f'Loaded data from {path}')
    return df
  except Exception as e:
    logger.error(f'Error reading csv: {e} ')


def pickle_reader(path: str =None) -> pd.DataFrame:
  """
  Function to read a dataset stored in Pickle format.
  
  Args:
    path: str = None -> Path to the dataset stored as Pickle.
    
  Return:
    df: pd.DataFrame -> Returns the data as a dataframe.
  """
  try:
    with open(path, 'rb') as file:
      df = pickle.load(file)
    logger.info(f'Loaded data from {path}.')
    return df
  except Exception as e:
    logger.error(f'Error reading pickle file: {e}')
    


def csv_read_and_convert(path: str = None, column_names_to_convert: list =None) -> pd.DataFrame:  
  """
  Function to read the dataset and convert the tokenized columns from string to list format.
  
  Args:
    path: str = None -> Path to the dataset stored as CSV.
    column_names_to_convert: list = None -> List containing the names of the columns to be converted.
    
  Return:
    df: pd.DataFrame -> Returns the data as a dataframe.  
  """
  try:
    df = pd.read_csv(path)
    
    if column_names_to_convert:
      for column in column_names_to_convert:
        if column in df.columns:
          df[column] = df[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        else:
          logger.warning(f'Column {column} not in dataset.')
    return df
  except Exception as e:
    logger.error(f'Error reading or converting the csv: {e}')
  

def txt_reader(dir: str =None) -> pd.DataFrame:
  """
  Searches in a directory for txt-files and converts them to a dataset
  
  Args:
    dir: str =None -> Path to the directory where the txt-files are located.
    
  Return:
    df:  pd.DataFrame -> Returns the data as a dataframe.
  
  """
  try:
    data = []
    
    for subdir in os.listdir(dir):
      subdir_path = os.path.join(dir, subdir)
      
      for file in os.listdir(subdir_path):
        if file.endswith('.txt'):
          result = file_info_resolver(dir_path=subdir_path, file=file)
          if result is not None:
            data.append(result)

    df = pd.DataFrame(data, columns=['id', 'content', 'rating'])
  
    logger.info(f'Found {len(df)} txt files.')
    return df
  
  except Exception as e:
    logger.error(f'Error reading the txt files: {e}')
    df = pd.DataFrame(columns=['id', 'content', 'rating'])
    return df
    
    
def file_info_resolver(dir_path: str, file: str) -> tuple:
  """
  Extracts the information from the txt file.
  
  Args:
    dir_path: str -> Path to the directory containing the txt-files.
    file: str -> File where the information is extracted from.
    
  Return:
    :tuple -> Extracted information.  
  """
  try:
    file_path = os.path.join(dir_path, file)
    file_name = file.rsplit('.', 1)[0]
    file_id, file_rating = file_name.rsplit('_', 1)
    
    with open(file_path, 'r', encoding='utf-8') as f:
      description = f.read().strip()
    logger.info(f'Extracted info from {file}')
    return file_id, description, file_rating
  except Exception as e:
    logger.warning(f'Unable to extract information from file {file}: {e}')
    return None


def read_stanford_data(path: str, train_test: bool):
  """
  For multithreading of reading the txt files.
  Args:
    path: str -> Path to the directory with the data.
    train_test: bool -> if True saved as training data, else as test data.
  """
  stanford_df = txt_reader(path)
  if train_test:
    stanford_df.to_csv('data/raw/stanford_train_data.csv', index=False)
  else:
    stanford_df.to_csv('data/raw/stanford_test_data.csv', index=False)
  
# Run only for extracting the information from the stanford dataset.
# Convert information into CSV-files for training and testing.
# Stanford data is not included in this package.
# Download from: https://ai.stanford.edu/~amaas/data/sentiment/
# Requires directory structure:
# dir
# |--- subidr
#       |--- id_rating.txt
# ...

if __name__=='__main__':
  default_path_to_stanford_training_data = 'data/raw/stanford_dataset/train'
  default_path_to_stanford_test_data = 'data/raw/stanford_dataset/test' 
  
  stanford_train_data_path = input(f'Enter the path to the Stanford training data [{default_path_to_stanford_training_data}]: ').strip() or default_path_to_stanford_training_data
  stanford_test_data_path = input(f'Enter the path to the Stanford test data [{default_path_to_stanford_test_data}]: ').strip() or default_path_to_stanford_test_data
  
  train_data_thread = threading.Thread(target=read_stanford_data, args=(stanford_train_data_path, True,), daemon=True)
  test_data_thread = threading.Thread(target=read_stanford_data, args=(stanford_test_data_path, False,), daemon=True)
  
  train_data_thread.start()
  test_data_thread.start()
  
  train_data_thread.join()
  test_data_thread.join()
  