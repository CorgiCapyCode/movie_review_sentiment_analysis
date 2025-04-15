import logging
import pandas as pd
from src.preprocessing.read_data import csv_reader
from src.preprocessing.preprocessing import apply_preprocessing
from src.preprocessing.vectorization import train_word2vec, load_pretrained_word2vec, apply_word2vec, apply_roberta
from src.training.training import create_all_models

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
  
  print('Finished preprocessing.')
  print(df_preprocessed.info())
  
  # Word Vectorization
  print('Starting vectorization.')
  columns_with_tokens = [
        'tokenized',
        'no_stopwords',
        'stemmed_no_sw',
        'stemmed',
        'lemmatized',
        'lemmatized_no_sw',
    ]
  
  print('   Start training Word2Vec...')
  train_word2vec(df=df_preprocessed, token_column='tokenized', vector_size=300, window=5)
  print('   Finished the training.')
    
  df_for_google_news = df_preprocessed.copy()
  df_for_self_trained = df_preprocessed.copy()
  df_for_roberta = df_preprocessed.copy()
  google_model = load_pretrained_word2vec(model_path='src/preprocessing/vec_model_weights/GoogleNews-vectors-negative300.bin')
  self_trained_model = load_pretrained_word2vec(model_path='src/preprocessing/vec_model_weights/self_trained_word2vec.bin')
  
  print(' Start vectorization for Word2Vec models...')
  df_with_google_vectors = apply_word2vec(df=df_for_google_news, token_columns=columns_with_tokens, model=google_model,  output_name='google_news_word2vec', drop_tokens=True)
  df_with_self_trained_vectors = apply_word2vec(df=df_for_self_trained, token_columns=columns_with_tokens, model=self_trained_model,  output_name='self_trained_word2vec', drop_tokens=True)
  
  print('Start vectorization with RoBERTa...')
  df_with_roberta = apply_roberta(df=df_for_roberta, token_columns=columns_with_tokens, use_cuda=True, output_name='roberta_vecs', drop_tokens=True)
  
  # Model Training
  label_column = 'sentiment_binary'
  
  feature_columns = [
    'review_vector_tokenized',
    'review_vector_no_stopwords',
    'review_vector_stemmed_no_sw',
    'review_vector_stemmed',
    'review_vector_lemmatized',
    'review_vector_lemmatized_no_sw'    
    ]
  
  models = {
        'training_setup' : {
            'test_size' : 0.99,
            'random_state' : 17
        },
        'SVM': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale'
        },
        'RF': {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': 5,
            'random_state': 17
        },
        'ANN': {
            'hidden_units': [128, 64],
            'activation': 'relu',
            'learning_rate': 0.001,
            'epochs': 30,
            'batch_size': 64,
            'dropout': 0.3,
            'optimizer': 'adam',
            'regularization': {'l2': 0.001},
            'early_stopping': True,
        }
    }
  
  dataset_paths = {
        'google_word2vec': 'data/vectorized/google_news_word2vec.pkl',
        'self_trained_word2vec': 'data/vectorized/self_trained_word2vec.pkl',
        'roberta' : 'data/vectorized/roberta_vecs.pkl'   
    }
  
  print('Start training the models.')
  create_all_models(
        training_params=models,
        dataset_paths=dataset_paths,
        label_column=label_column,
        feature_columns=feature_columns,
        output_path='data/training/models',
        results_path='data/training'
    )