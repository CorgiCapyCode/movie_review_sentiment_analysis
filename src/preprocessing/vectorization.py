import functools
import gensim
import logging
import numpy as np
import os
import pandas as pd
import threading
import torch

from gensim.models import KeyedVectors, Word2Vec
from read_data import pickle_reader
from transformers import RobertaTokenizer, RobertaModel

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Word2Vec
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
    

def apply_word2vec(df: pd.DataFrame, token_columns: list, model, output_name: str, drop_tokens: bool =False) -> pd.DataFrame:
    """
    Applies Word2Vec on the tokenized columns.
    
    Args:
        df: pd.Dataframe -> DataFrame containing the word tokens.
        token_columns: list -> Name of the columns with tokens.
        model -> The Word2Vec model.
        output_name: str -> THe name of the file containing the results. Stored under: data/vectorized
        drop_tokens: bool =True -> Defines whether the original tokens are dropped and only the word vectors are kept (True). Reduces the output file size.
    
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
        
        if drop_tokens:
            df.drop(columns=token_columns, inplace=True)
            logger.info('Token columns dropped successfully.')
            
        df.to_csv(f'data/vectorized/{output_name}.csv')
        df.to_pickle(f'data/vectorized/{output_name}.pkl')
        return df
    
    except ValueError as v:
        logger.error(f'ValueError: {v}')
    except TypeError as t:
        logger.error(f'TypeError: {t}')
    except Exception as e:
        logger.error(f'Error while applying Word2Vec: {e}')
  

results = {}


def run_word2Vec_task( key: str, df: pd.DataFrame, token_columns: list, model, output_name: str, drop_tokens: bool =False) -> pd.DataFrame:
    """
    Applies Word2Vec on the tokenized columns. Stores the results in the global results.
    
    Args:
        key: str -> Stores the results under the given name.
        df: pd.Dataframe -> DataFrame containing the word tokens.
        token_columns: list -> Name of the columns with tokens.
        model -> The Word2Vec model.
        output_name: str -> THe name of the file containing the results. Stored under: data/vectorized
        drop_tokens: bool =True -> Defines whether the original tokens are dropped and only the word vectors are kept (True). Reduces the output file size.
    """
    results[key] = apply_word2vec(df=df, token_columns=token_columns, model=model, output_name=output_name, drop_tokens=drop_tokens)
    

# RoBERTa
@functools.lru_cache(maxsize=1)
def load_roberta():
    """
    Loads the RoBERTa model and tokenizer.
    """
    try:
        # The tokenizer is required since RoBERTa requires a special tokenization in order to avoid issues.
        # The tokenizer is applied on the tokenized columns in order to also maintain the effects from e.g. stop-word removal.
        roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        roberta_model = RobertaModel.from_pretrained('roberta-base')
        logger.info('Loaded the RoBERTa tokenizer and model.')
        return roberta_tokenizer, roberta_model
    except Exception as e:
        logger.error(f'Error loading RoBERTa model or tokenizer: {e}')
        return None, None


def get_roberta_vector(tokens: list, use_cuda: bool =True) -> np.ndarray:
    """
    Generates a vector for the review based on the RoBERTa model.
    Args:
        tokens: list -> The input review.
        use_cuda: bool =True -> Enables CUDA if true.
        
    Returns:
        np.ndarray -> Review vector based on RoBERTa.
    """
    
    try:
        roberta_tokenizer, roberta_model = load_roberta()
        if roberta_tokenizer is None or roberta_model is None:
            return np.zeros(768)
        
        if use_cuda and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        
        roberta_model.to(device)
        
        review = " ".join(tokens)
        
        inputs = roberta_tokenizer(review, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = roberta_model(**inputs)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        
        logger.info("Embedded successfully")
        return cls_embedding
    except Exception as e:
        logger.error(f'Error embedding the review: {e}')
        return np.zeros(768)


def apply_roberta(df: pd.DataFrame, token_columns: list, use_cuda: bool =True, output_name: str ='roberta_vec', drop_tokens: bool =True) -> pd.DataFrame:
    """
    Applies RoBERTa on the tokenized columns.
    
    Args:
        df: pd.Dataframe -> DataFrame containing the word tokens.
        token_columns: list -> Name of the columns with tokens.
        use_cuda: bool =True -> Enables CUDA if true.
        output_name: str -> THe name of the file containing the results. Stored under: data/vectorized
        drop_tokens: bool =True -> Defines whether the original tokens are dropped and only the word vectors are kept (True). Reduces the output file size.
    
    Returns:
        pd.DataFrame -> DataFrame including a new column for a review vector.
    """
    try:
        for token_column in token_columns:
            if token_column not in df.columns:
                raise ValueError(f'Column: {token_column} not found in DataFrame.')

            if not all(isinstance(tokens, list) for tokens in df[token_column]):
                raise TypeError(f'Column: {token_column} does not contain valid lists of tokens.')

            df[f"review_vector_{token_column}"] = df[token_column].apply(lambda tokens: get_roberta_vector(tokens, use_cuda))

        if drop_tokens:
            df.drop(columns=token_columns, inplace=True)
            logger.info('Token columns dropped successfully.')

        df.to_pickle(f"data/vectorized/{output_name}.pkl")
        df.to_csv(f"data/vectorized/{output_name}.csv")

        logger.info("RoBERTa embeddings applied successfully.")
        return df

    except ValueError as v:
        logger.error(f"ValueError: {v}")
    except TypeError as t:
        logger.error(f"TypeError: {t}")
    except Exception as e:
        logger.error(f"Error while applying RoBERTa: {e}")    
 
 
if __name__=='__main__':
    # Define which vectorization method should be applied.
    run_word2vec = True
    run_RoBERTa = True
    
    columns_with_tokens = [
        'tokenized',
        'no_stopwords',
        'stemmed_no_sw',
        'stemmed',
        'lemmatized',
        'lemmatized_no_sw',
    ]
    
    print('Start loading the dataset.')
    #df = csv_read_and_convert(path='data/preprocessed/preprocessed_dataset.csv', column_names_to_convert=columns_with_tokens)
    df = pickle_reader(path='data/preprocessed/preprocessed_dataset.pkl')
    print('Finished loading the dataset.')

    if run_word2vec:
        # For training of the word2vec model.
        print('Start the training process for Word2Vec based on the reviews.')
        train_word2vec(df=df, token_column='tokenized', vector_size=300, window=5)
        print('Finished the training.')

        print('Start Word2Vec vectorization with the model from GoogleNews and the self trained model.')
        df_for_google_news = df.copy()
        df_for_self_trained = df.copy()    
        google_model = load_pretrained_word2vec(model_path='src/preprocessing/vec_model_weights/GoogleNews-vectors-negative300.bin')    
        self_trained_model = load_pretrained_word2vec(model_path='src/preprocessing/vec_model_weights/self_trained_word2vec.bin')
        
        if google_model is None or self_trained_model is None:
            logger.error('failed to load min. one of the models. Execution stopped.')
            exit(1)
        
        google_thread = threading.Thread(target=run_word2Vec_task, args=('google_news', df_for_google_news, columns_with_tokens, google_model, 'google_news_word2vec', True))
        self_trained_thread = threading.Thread(target=run_word2Vec_task, args=('self_trained', df_for_self_trained, columns_with_tokens, self_trained_model, 'self_trained_word2vec', True))
        
        google_thread.start()
        self_trained_thread.start()
        
        google_thread.join()
        self_trained_thread.join()
        
        df_with_google_news = results.get('google_news')
        df_with_self_trained = results.get('self_trained')
        
        print('Finished processing Word2Vec.')
        
        print('Results from Google News:')
        print(df_with_google_news.info())
        print('Results from Self-Trained:')
        print(df_with_self_trained.info())        
        
    # Warning: The task is computational intensive.
    if run_RoBERTa:
        print('Start RoBERTa vectorization based on the pretrained model. WARNING: THis might take a while...')
        df_for_roberta = df.copy()
        df_with_roberta = apply_roberta(df=df_for_roberta, token_columns=columns_with_tokens, use_cuda=True, output_name='roberta_vecs', drop_tokens=True)
        print('Finished the vectorization with RoBERTa.')
        print('Results from Roberta:')
        print(df_with_roberta.info())
        
    
    