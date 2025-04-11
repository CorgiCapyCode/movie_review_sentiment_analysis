import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow

from src.preprocessing.read_data import pickle_reader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def create_all_models(training_params: dict, dataset_paths: dict,  label_column: str, feature_columns: list, test_size: float =0.2, random_state: int =17, output_path: str ='data/training/models', results_path: str ='data/training'):
    """
    Trains and tests the specified models on the selected datasets.
    
    Args:
        training_params: dict -> Dictionary of the model names that will be trained including their hyperparameters. Available: SVM, RF, ANN.
        dataset_paths: dict -> Dictionary containing the paths to the datasets.
        label_column: str -> Name of the column with the labels.
        feature_columns: list -> Names of the columns containing the vectorized data.
        output_path: str ='data/training/models' -> Location where the results are stored.
        results_path: str ='data/training' -> Location where the results are stored.
    """
    try:
        datasets = {}
        test_results = {}
        
        for dataset_name, dataset_path in dataset_paths.items():
            df = pickle_reader(dataset_path)
            datasets[dataset_name] = df
        
        #print('Preparing datasets...')
        datasets = filter_columns(datasets=datasets, feature_columns=feature_columns, label_column=label_column)
        #print('...33%...')
        datasets = split_datasets(datasets=datasets, training_setup=training_params['training_setup'])
        #print('...66%...')
        datasets = pair_features_with_labels(datasets=datasets, feature_columns=feature_columns, label_column=label_column)
        #print('Preparation completed.')

        # Only to check the structure for of the dataset.
        #datachecker_after_conversion(datasets=datasets)
    

        if 'SVM' in training_params:
            print('Start SVM model trainings.')
            test_results['SVM'] = orchestrate(model_type='SVM', dataset_dict=datasets, hyperparameter= training_params['SVM'], output_path=output_path)
            print('SVM results')
            print(test_results['SVM'])
            print('-' * 50)
        
        if 'RF' in training_params:
            print('Start Decision Tree model trainings.')
            test_results['RF'] = orchestrate(model_type='RF', dataset_dict=datasets, hyperparameter= training_params['RF'], output_path=output_path)
            print('RF results')
            print(test_results['RF'])
            print('-' * 50)
            
        if 'ANN' in training_params:
            print('Start ANN model training.')
            test_results['ANN'] = orchestrate(model_type='ANN', dataset_dict=datasets, hyperparameter= training_params['ANN'], output_path=output_path)
            print('ANN results')
            print(test_results['ANN'])
            print('-' * 50)
        
        save_results_as_json(results_path, test_results, training_params)

        results_file = os.path.join(results_path, 'results_and_hyperparams.pkl')
        with open(results_file, 'wb') as file:
            pickle.dump({'test_results': test_results, 'hyperparameters': training_params}, file)        
    
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
        return None


def split_datasets(datasets: dict, training_setup: dict) -> dict:
    """
    Function to split the datasets into training and testing.
    
    Args:
        datasets: dict -> Dictionary with the datasets.
        training_setup: dict-> Dictionary containing the necessary information for training.
        
    Returns:
        dict -> Dictionary with a training and testing dataset as value.
    """
    try:
        test_size = training_setup['test_size']
        random_state = training_setup['random_state']
        return {name: train_test_split(df, test_size=test_size, random_state=random_state) for name, df in datasets.items()}
    except Exception as e:
        logger.error(f'Error while splitting the datasets: {e}')
        return None


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
        return None


def datachecker_after_conversion(datasets):
    for key, ((X_train, y_train), (X_test, y_test)) in datasets.items():
        print(f'{key}:')
        print(f'Train - Features Shape: {X_train.shape}, Labels Shape: {y_train.shape}')
        print(f'Test  - Features Shape: {X_test.shape}, Labels Shape: {y_test.shape}')
        print("-" * 50)    


def orchestrate(model_type: str, dataset_dict: dict, hyperparameter: dict, output_path: str ='data/training/models'):
    """
    Orchestrates the training of the SVM or DT models. Saves the models.
    
    Args:
        model_type: str -> The model type to be trained: SVM, RF or ANN
        dataset_dict: dict ->  Dictionary with the data.
        hyperparameters: dict -> Dictionary with the hyperparameters needed for training.
        output_path: str ='data/training/models' -> Path to the directory where the models are stored.
        
    Returns:
        test_results: dict -> All test results.
    """
    try:
        test_results = {}
        
        for name, (train_data, test_data) in dataset_dict.items():
            print(f'Name of the model to be trained: {name}')
            if model_type=='SVM':
                model = train_svm(X_train=train_data[0], y_train=train_data[1], hp=hyperparameter)
                save_svm_rf_model(model=model, output_path=output_path, name=f'{name}_svm')
            elif model_type=='RF':
                model = train_rf(X_train=train_data[0], y_train=train_data[1], hp=hyperparameter)
                save_svm_rf_model(model=model, output_path=output_path, name=f'{name}_rf')
            elif model_type=='ANN':
                model = train_ann(X_train=train_data[0], y_train=train_data[1], hp=hyperparameter)
                save_ann_model(model=model, output_path=output_path, name=f'{name}_ann')
            else:
                logger.error('Invalid model selected.')
                return None
            
            results = test_model(X_test=test_data[0], y_test=test_data[1], model=model)
            test_results[name] = results
            logger.info('Model training and testing for all models completed.')
        
        return test_results

    except Exception as e:
        logger.error(f'Error while training or testing: {e}')
        return None
    

#kernel: str ='rbf', C: float =1.0
def train_svm(X_train: np.ndarray, y_train: np.ndarray, hp: dict) -> SVC:
    """
    Trains a SVM model as specified.
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
        return None


def train_rf(X_train: np.ndarray, y_train: np.ndarray, hp: dict):
    """
    Trains a Random Forest model as specified.
    Hyperparameters:
        n_estimators: int -> Number of trees in the forest.
        criterion: str -> Quality measurement,  default value: 'gini'
        max_depth: int or None -> Maximum depth of the tree, default value: None
    
    Args:
        X_train: np.ndarray -> The data array with the training features.
        y_train: np.ndarray -> The data array with the training labels.
        hp: dict -> Dictionary containing the hyperparameters.
    
    Returns:
        model
    """
    try:
        X_train = np.vstack(X_train[:, 0])
        model = RandomForestClassifier(**hp)
        model.fit(X_train, y_train)
        logger.info('RandomForestClassifier trained.')
        return model
    except Exception as e:
        logger.error(f'Error while training the random forest" {e}')
        return None


def train_ann(X_train: np.ndarray, y_train: np.ndarray, hp: dict):
    """
    Trains a simple ANN model as specified.
    
    Args:
        X_train: np.ndarray -> The data array with the training features.
        y_train: np.ndarray -> The data array with the training labels.
        hp: dict -> Dictionary containing the hyperparameters.
    
    Returns:
        model
    """
    try:
        X_train = np.vstack(X_train[:, 0])
        
        
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        hidden_units = hp.get('hidden_units', [16])
        
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
            
        reg_value = hp.get('regularization', {}).get('l2', 0.0)
            
        for units in hidden_units:
            model.add(Dense(units, activation=hp.get('activation', 'relu'), kernel_regularizer=l2(reg_value)))
            
            if hp.get('dropout', 0) > 0:
                model.add(Dropout(hp['dropout']))
                
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(learning_rate=hp.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = []
        if hp.get('early_stopping', False):
            callbacks.append(EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True))

        model.fit(X_train, y_train, 
                  epochs=hp.get('epochs', 10), 
                  batch_size=hp.get('batch_size', 32), 
                  validation_split=0.2,
                  callbacks=callbacks, 
                  verbose=1)

        logger.info('ANN model trained.')
        return model

    except Exception as e:
        logger.error(f'Error while training ANN: {e}')
        return None


def test_model(X_test: np.ndarray, y_test: np.ndarray, model):
    """
    Calculates the test metrics: Accuracy, Precision, Recall, F1-Score and the confusion matrix.
    
    Args:
        X_test: np.ndarray -> The data array with the testing features.
        y_test: np.ndarray -> The data array with the testing labels.
        model -> The model, which is tested.
        
    Returns:
        results: dict -> Dictionary containing the test results.
    """
    try:
        X_test = np.vstack(X_test[:, 0])
        #print(f"X_test shape: {X_test.shape}, type: {type(X_test)}")
        #print(f"y_test shape: {y_test.shape}, type: {type(y_test)}")
        
        predictions = model.predict(X_test)
        binary_predictions = (predictions > 0.5).astype(int)
        
        results = {
            'accuracy' : accuracy_score(y_test, binary_predictions),
            'f1_score' : f1_score(y_test, binary_predictions, average='weighted'),
            'recall' : recall_score(y_test, binary_predictions, average='weighted'),
            'precision' : precision_score(y_test, binary_predictions, average='weighted'),
            'confusion_matrix' : confusion_matrix(y_test, binary_predictions).tolist()
        }
        return results
    except Exception as e:
        logger.error(f'Error while testing: {e}')
        return None


def save_svm_rf_model(model, output_path: str, name: str):
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


def save_ann_model(model, output_path: str, name: str):
    """
    Saves the ANN model under the given path in HDF5 format.
    
    Args:
        model -> The ANN model to be saved.
        output_path: str -> Location where the model will be stored.
        name -> file name of the model.
    """
    try:
        os.makedirs(output_path, exist_ok=True)
        model_filename = os.path.join(output_path, f"{name}.keras")
        
        model.save(model_filename)
        logger.info(f'ANN model saved at {model_filename}')
        
    except Exception as e:
        logger.error(f'Error while saving the model: {e}')
 

def save_results_as_json(results_path: str, test_results: dict, training_params: dict):
    """
    Saves test results and hyperparameters as JSON.
    
    Args:
        results_path: str -> Path to store the results.
        test_results: dict -> Dictionary of test results.
        training_params: dict -> Dictionary of model hyperparameters.
    """
    try:
        results_file = os.path.join(results_path, 'results_and_hyperparams.json')

        results_data = {
            'test_results': test_results,
            'training_params': training_params
        }

        with open(results_file, 'w') as file:
            json.dump(results_data, file, indent=4)

        print(f"Results successfully saved in {results_file}")

    except Exception as e:
        logger.error(f"Error while saving results: {e}")       


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
        'training_setup' : {
            'test_size' : 0.9,
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



    ### TEST SECTION ONLY:

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
        'training_setup' : {
            'test_size' : 0.9,
            'random_state' : 17
        },
        'RF': {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': 5,
            'random_state': 17
        }
    }




    create_all_models(
        training_params=models,
        dataset_paths=dataset_paths,
        label_column=label_column,
        feature_columns=feature_columns,
        output_path='data/training/models',
        results_path='data/training'
    )
