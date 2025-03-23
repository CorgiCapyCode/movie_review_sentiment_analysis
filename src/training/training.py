import joblib
import logging
import numpy as np
import os
import pandas as pd
import pickle

from src.preprocessing.read_data import pickle_reader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def create_all_models(model_list: dict, dataset_paths: dict, label_column: str, feature_columns: list, test_size: float =0.2, random_state: int =17, output_path: str ='data/training/models'):
    """
    Trains and tests the specified models on the selected datasets.
    
    Args:
        model_list: dict -> Dictionary of the model names that will be trained including their hyperparameters. Available: SVM, DT, ANN.
        dataset_paths: dict -> Dictionary containing the paths to the datasets.
        label_column: str -> Name of the column with the labels.
        feature_columns: list -> Names of the columns containing the vectorized data.
        test_size: float =0.2 -> Defining the ratio between test and train data.
        random_state: int =17 -> Defines the random state for reproducibility.
    """
    try:
        datasets = {}
        
        for dataset_name, dataset_path in dataset_paths.items():
            df = pickle_reader(dataset_path)
            datasets[dataset_name] = df
        
        #print('Preparing datasets...')
        datasets = filter_columns(datasets=datasets, feature_columns=feature_columns, label_column=label_column)
        #print('...33%...')
        datasets = split_datasets(datasets=datasets, test_size=test_size, random_state=random_state)
        #print('...66%...')
        datasets = pair_features_with_labels(datasets=datasets, feature_columns=feature_columns, label_column=label_column)
        #print('Preparation completed.')

        # Only to check the structure for of the dataset.
        #datachecker_after_conversion(datasets=datasets)
    

        if 'SVM' in model_list:
            #print('Start SVM model training')
            svm_test_results = orchestrate_svm(dataset_dict=datasets, hyperparameter= model_list['SVM'], output_path=output_path)
            
            print('Results of the SVM model trainings:')
            print(svm_test_results)
        
        if 'DT' in model_list:
            # Run Decision Tree training and testing
            pass
        
        if 'ANN' in model_list:
            # Run ANN training and testing
            pass
    
    except Exception as e:
        logger.error(f'Error while running training and testing of all models" {e}')


# Dataset preparation
def filter_columns(datasets: dict, feature_columns: list, label_column: str) -> dict:
    """
    Filters for only the relevant features which are considered for the model creation.
    
    Args:
        datasets: dict -> Dictionary with the datasets.
        label_column: str -> Name of the column with the labels.
        feature_columns: list -> Names of the columns containing the vectorized data.
        
    Returns:
        dict -> Dictionary with the datasets, but with dataframes containing only the relevant columns.
    """
    try:
        relevant_columns = feature_columns + [label_column]
        return {name: df[relevant_columns] for name, df in datasets.items()}
    except Exception as e:
        logger.error(f'Error while filtering for relevant columns: {e}')


def split_datasets(datasets: dict, test_size: float =0.2, random_state: int =17) -> dict:
    """
    Function to split the datasets into training and testing.
    
    Args:
        datasets: dict -> Dictionary with the datasets.
        test_size: float =0.2 -> Defining the ratio between test and train data.
        random_state: int =17 -> Defines the random state for reproducibility.
        
    Returns:
        dict -> Dictionary with a training and testing dataset as value.
    """
    try:
        return {name: train_test_split(df, test_size=test_size, random_state=random_state) for name, df in datasets.items()}
    except Exception as e:
        logger.error(f'Error while splitting the datasets: {e}')


def pair_features_with_labels(datasets: dict, feature_columns: list, label_column: str) -> dict:
    """
    Generates pairs of the features with labels that are then used for training.
    
    Args:
        datasets: dict -> Dictionary with the datasets.
        label_column: str -> Name of the column with the labels.
        feature_columns: list -> Names of the columns containing the vectorized data.
    
    Returns:
        dict -> Dictionary with one entry for each feature-label pair.
    """
    try:
        feature_label_pair = {}
        for name, (train_df, test_df) in datasets.items():
            for feature in feature_columns:
                train_pair = (train_df[[feature]].to_numpy(), train_df[label_column].to_numpy())
                test_pair = (test_df[[feature]].to_numpy(), test_df[label_column].to_numpy())
                feature_label_pair[f'{name}_{feature}'] = (train_pair, test_pair)
        logger.info('Converted the data successfully.')
        return feature_label_pair
    except Exception as e:
        logger.error(f'Error while converting the data: {e}')


def datachecker_after_conversion(datasets):
    for key, ((X_train, y_train), (X_test, y_test)) in datasets.items():
        print(f"{key}:")
        print(f"  Train - Features Shape: {X_train.shape}, Labels Shape: {y_train.shape}")
        print(f"  Test  - Features Shape: {X_test.shape}, Labels Shape: {y_test.shape}")
        print("-" * 50)    

# SVM training and testing

def orchestrate_svm(dataset_dict: dict, hyperparameter: dict, output_path: str ='data/training/models'):
    """
    Orchestrates the training of the SVM models. Saves the models.
    
    Args:
        dataset_dict: dict ->  Dictionary with the data.
        hyperparameters: dict -> Dictionary with the hyperparameters needed for training of the SVM.
        output_path: str ='data/training/models' -> Path to the directory where the models are stored.
        
    Returns:
        test_results: dict -> All test results of the SVM models created.
    """
    try:
        test_results = {}
        for name, (train_data, test_data) in dataset_dict.items():
            #print(f'Name of the model to be trained: {name}')
            model = train_svm(X_train=train_data[0], y_train=train_data[1], hp=hyperparameter)
            results = test_model(X_test=test_data[0], y_test=test_data[1], model=model)
            
            save_svm_dt_model(model=model, output_path=output_path, name=name)
            test_results[name] = results
            logger.info('Model training and testing for all SVMs completed.')
        
        return test_results

    except Exception as e:
        logger.error(f'Error while training or testing: {e}')
    

#kernel: str ='rbf', C: float =1.0
def train_svm(X_train: np.ndarray, y_train: np.ndarray, hp: dict) -> SVC:
    """
    Trains an SVM model as specified.
    Hyperparameters:
        kernel: str -> SVC default value: 'rbf'
        C: float -> SVC default value: 1.0
    
    Args:
        X_train: np.ndarray -> The data array with the training features.
        y_train: np.ndarray -> The data array with the training labels.
        hp: dict -> Dictionary containing the hyperparameters.
    
    Returns:
        model
    """
    try:
        X_train = np.vstack(X_train[:, 0])
        #print(f"X_train shape: {X_train.shape}, type: {type(X_train)}")
        #print(f"y_train shape: {y_train.shape}, type: {type(y_train)}")


        model =  SVC(**hp)
        model.fit(X_train, y_train)
        logger.info('SVM model trained.')
        return model
    except Exception as e:
        logger.error(f'Error while training model: {e}')


def test_model(X_test: np.ndarray, y_test: np.ndarray, model):
    try:
        X_test = np.vstack(X_test[:, 0])
        #print(f"X_test shape: {X_test.shape}, type: {type(X_test)}")
        #print(f"y_test shape: {y_test.shape}, type: {type(y_test)}")
        
        predictions = model.predict(X_test)
        return accuracy_score(y_test, predictions)
    except Exception as e:
        logger.error(f'Error while testing: {e}')


def save_svm_dt_model(model, output_path: str, name: str):
    """
    Saves the model under the given path.
    
    Args:
        model -> The model to be saved.
        output_path: str -> Location where the model will be stored.
        name -> file name of the model.
    """
    try:
        os.makedirs(output_path, exist_ok=True)
        model_filename = os.path.join(output_path, f"{name}.pkl")
        
        with open(model_filename, "wb") as file:
            pickle.dump(model, file)
        
    except Exception as e:
        logger.error(f'Error while saving the model: {e}')


if __name__=='__main__':
    
    dataset_paths = {
        'google_word2vec': 'data/vectorized/google_news_word2vec.pkl',
        'self_trained_word2vec': 'data/vectorized/self_trained_word2vec.pkl',
        'roberta' : 'data/vectorized/roberta_vecs.pkl'   
    }
    
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
        'SVM': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale'
        }
    }
    
    # For testing with only one dataset and feature
    dataset_paths = {
        'google_word2vec': 'data/vectorized/google_news_word2vec.pkl' 
    }
    feature_columns = [
        'review_vector_tokenized',
        'review_vector_no_stopwords',
        'review_vector_stemmed_no_sw',
        'review_vector_stemmed',
        'review_vector_lemmatized',
        'review_vector_lemmatized_no_sw'   
    ]
    
    create_all_models(model_list=models, dataset_paths=dataset_paths, label_column=label_column, feature_columns=feature_columns)
