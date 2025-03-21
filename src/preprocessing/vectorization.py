import gensim
import logging
import numpy as np
import os
import pandas as pd
import threading

from gensim.models import KeyedVectors, Word2Vec
from read_data import csv_read_and_convert

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def load_pretrained_word2vec(model_path: str ='src/preprocessing/vec_model_weights/GoogleNews-vectors-negative300.bin'):
    """
    Load pre-trained Word2Vec model.
    Args:
        model_path: str = 'src/preprocessing/vec_model_weights/GoogleNews-vectors-negative300.bin' -> The path to the model weights for vectorization.
    
    Return:
        pretrained_word2vec -> Model parameters for the pretrained Word2Vec
        
    Raises:
        FileNotFoundError -> If the model is not found.
    """
    
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file not found at: {model_path}')
        pretrained_word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)
        logger.info(f'Loaded model weights from {model_path}.')
        return pretrained_word2vec
    except FileNotFoundError as fnf:
        logger.error(f'FileNotFoundError: {fnf}')
    except Exception as e:
        logger.error(f'Error loading model: {e}')


def train_word2vec(df: pd.DataFrame, token_column: str, vector_size: int =300, window: int =5, model_path: str ='src/preprocessing/vec_model_weights') -> pd.DataFrame:
    """
    Trains Word2Vec using Skip-Gram based on the tokenized reviews.
    
    Args:
        df: pd.DataFrame -> Dataframe containing the word tokens.
        token_column: str -> Name of the column with the tokens.
        vector_size: int =300 -> Vector size of the word vectors.
        window: int =5 -> Context window size
        model_path: str = 'data/vectorized' -> Location where the model is stored.
        
    Returns:
        pd.DataFrame -> DataFrame with the word vectors attached.
        
    Raises:
        ValueError -> if the column name does not exist.
        TypeError -> if the column does not contain tokens.
    """
    
    try:
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            logger.warning(f'Model path {model_path} did not exist. Was created.')
        
        if token_column not in df.columns:
            raise ValueError(f'Column: {token_column} not found.')
        
        if not all(isinstance(tokens, list) for tokens in df[token_column]):
            raise TypeError(f'Column: {token_column} does not contain valid lists of tokens.')
        
        model = Word2Vec(
            sentences=df[token_column],
            vector_size=vector_size,
            window=window,
            sg=1,
            workers=4,
            epochs=10
        )
        model.wv.save_word2vec_format(f'{model_path}/self_trained_word2vec.bin', binary=True)

        logger.info('Word2Vec model trained and stored.')
    
    except ValueError as v:
        logger.error(f'ValueError: {v}')
    except TypeError as t:
        logger.error(f'TypeError: {t}')    
    except Exception as e:
        logger.error(f'Error while training: {e}')  


def get_review_vector(tokens: list, model) -> np.ndarray:
    """
    Generates a review vector using the average.
    
    Args:
        tokens: list -> Tokenized words of a review.
        model -> The Word2Vec model.
    
    Returns:
        np.ndarray -> Review vector.
    """
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
    

def apply_word2vec(df: pd.DataFrame, token_columns: list, model, output_name: str) -> pd.DataFrame:
    """
    Applies Word2Vec on a tokenized column.
    
    Args:
        df: pd.Dataframe -> DataFrame containing the word tokens.
        token_columns: list -> Name of the columns with tokens.
        model -> The Word2Vec model.
    
    Returns:
        pd.DataFrame -> DataFrame including a new column for a review vector.
    """
    try:
        for token_column in token_columns:
            if token_column not in df.columns:
                raise ValueError(f'Column: {token_column} not found in DataFrame.')

            if not all(isinstance(tokens, list) for tokens in df[token_column]):
                raise TypeError(f'Column: {token_column} does not contain valid lists of tokens.')

            df[f'review_vector_{token_column}'] = df[token_column].apply(lambda tokens: get_review_vector(tokens, model))
            logger.info('Word2Vec applied successfully.')
            
        df.to_csv(f'data/vectorized/{output_name}.csv')
        return df
    
    except ValueError as v:
        logger.error(f'ValueError: {v}')
    except TypeError as t:
        logger.error(f'TypeError: {t}')
    except Exception as e:
        logger.error(f'Error while applying Word2Vec: {e}')
    

# ELMo

# RoBERTa


if __name__=='__main__':
    columns_with_tokens = [
        'tokenized',
        'no_stopwords',
        'stemmed_no_sw',
        'stemmed',
        'lemmatized',
        'lemmatized_no_sw',
    ]
    
    df = csv_read_and_convert(path='data/preprocessed/preprocessed_dataset.csv', column_names_to_convert=columns_with_tokens)
    df_google_news = df.copy()
    df_self_trained = df.copy()
    # For training of the word2vec model.
    # train_word2vec(df=df, token_column='tokenized', vector_size=300, window=5)
    
    google_model = load_pretrained_word2vec(model_path='src/preprocessing/vec_model_weights/GoogleNews-vectors-negative300.bin')    
    self_trained_model = load_pretrained_word2vec(model_path='src/preprocessing/vec_model_weights/self_trained_word2vec.bin')
    
    if google_model is None or self_trained_model is None:
        logger.error('failed to load min. one of the models. Execution stopped.')
        exit(1)
    
    google_thread = threading.Thread(target=apply_word2vec, args=(df_google_news, columns_with_tokens, google_model, 'google_news_word2vec'))
    self_trained_thread = threading.Thread(target=apply_word2vec, args=(df_self_trained, columns_with_tokens, self_trained_model, 'self_trained_word2vec'))
    
    google_thread.start()
    self_trained_thread.start()
    
    google_thread.join()
    self_trained_thread.join()
    
    print('Finished processing.')
    