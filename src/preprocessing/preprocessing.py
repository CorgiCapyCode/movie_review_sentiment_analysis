import logging
import nltk
import pandas as pd
import re
import threading

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import TreebankWordTokenizer

try:
    from read_data import csv_reader
except ImportError:
    from src.preprocessing.read_data import csv_reader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

lock = threading.Lock()

def check_and_download_nltk_resources() -> None:
    """
    Downloads the resources needed for the preprocessing.
    """
    resources = ['stopwords', 'wordnet', 'averaged_perceptron_tagger_eng']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
            logger.info(f'Resource {resource} already downloaded.')
        except LookupError:
            print(f'Downloading {resource}...')
            nltk.download(resource)
            logger.info(f'Resource {resource} downloaded/updated.')


def nltk_to_wordnet_pos(tag: str) -> str:
    """
    Used for POS-tagging in preparation for lemmatization.
    
    Args:
        tag: str -> POS-tag returned by pos_tag.
    
    Return:
        str -> The WordNet-POS tag.
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    

def apply_treebank_tokenization(df: pd.DataFrame, review_column: str ='review') -> pd.DataFrame:
    """
    Applies Treebank Word Tokenization.
    
    Args:
        df: pd.DataFrame -> The dataframe containing the reviews.
        review_column: str ='review' -> The target column containing the reviews.
        
    Return:
        df: pd.DataFrame -> Dataframe containing an additional column with tokens.
    
    """
    try:
        tokenizer = TreebankWordTokenizer()
        df["tokenized"] = df[review_column].apply(tokenizer.tokenize)
        logger.info('Finished tokenization')
        return df
    except Exception as e:
        logger.error(f'Error while tokenization: {e}')


def apply_stopword_removal(df: pd.DataFrame, tokenized_column:str = 'tokenized') -> pd.DataFrame:
    """
    Applies stop word removal.
    
    Args:
        df: pd.DataFrame -> Dataframe containing tokens of the reviews.
        tokenized_column: str = 'tokenized' -> Column containing the tokens.
    
    Return:
        df: pd.DataFrame -> Dataframe containing an additional column with removed stop words.    
    """
    try:
        stop_words = set(stopwords.words('english'))
        df['no_stopwords'] = df[tokenized_column].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])
        logger.info('Removed stop words.')
        return df
    except Exception as e:
        logger.error(f'Error while removing stop words: {e}')
        

def apply_lemmatization(df: pd.DataFrame, tokenized_column: str, output_column: str) -> None:
    """
    Applies POS-tagging and lemmatization.
    
    Args:
        df: pd.DataFrame -> Dataframe containing tokens of the reviews.
        tokenized_column: str = 'tokenized' -> Column containing the tokens.
        output_column: str -> name of the output column.
    """
    try:
        
        lemmatizer = WordNetLemmatizer()
        with lock:
            df[output_column] = df[tokenized_column].apply(
                lambda tokens: [lemmatizer.lemmatize(word, pos=nltk_to_wordnet_pos(tag) or wordnet.NOUN) for word, tag in pos_tag(tokens)]
            )
        logger.info(f'Applied lemmatization on column {tokenized_column}')
    except Exception as e:
        logger.error(f'Error while lemmatizaiton of column {tokenized_column}: {e}')


def apply_stemming(df: pd.DataFrame, tokenized_column: str, output_column: str) -> None:
    """
    
    Applies Snowball Stemming.
    Args:
        df: pd.DataFrame -> Dataframe containing tokens of the reviews.
        tokenized_column: str = 'tokenized' -> Column containing the tokens.
        output_column: str -> name of the output column.
    """
    try:
        stemmer = SnowballStemmer('english')
        df[output_column] = df[tokenized_column].apply(lambda tokens: [stemmer.stem(word) for word in tokens])
        logger.info(f'Applied stemming on column {tokenized_column}')
    except Exception as e:
        logger.error(f'Error while stemming of column {tokenized_column}: {e}')


def positive_negative_converter(df: pd.DataFrame, sentiment_column: str):
    """
    Converts the positive/negative string classifier into a numerical classifier.
    
    Args:
        df: pd.DataFrame -> Dataframe containing the classifier.
        sentiment_column: str -> Name of the column containing the labels.
    """
    df['sentiment_binary'] = df[sentiment_column].apply(lambda x: 1 if x == 'positive' else 0)


def remove_br_tags(df: pd.DataFrame, review_column: str ='review') -> pd.DataFrame:
    """
    Removes the <br /> tags from the text.
    
    Args:
        df: pd.DataFrame -> Dataframe containing the reviews.
        review_column: str = 'review' -> Name of the column containing the reviews.    
    
    Returns
        pd.Dataframe -> Dataset without the tags.
    """
    try:
        pattern = r'<br\s*/?>'  # for <br>, <br/> and <br /> tags
        removed_tags = df[review_column].str.count(pattern, flags=re.IGNORECASE).sum()
        df[review_column] = df[review_column].str.replace(pattern, ' ', flags=re.IGNORECASE, regex=True)
        logger.info(f'Removed {removed_tags} tags from the text.')
        print(f'Tags removed: {removed_tags}')
        return df        
    except Exception as e:
        logger.error(f'Error while removing <br /> tags: {e}')
        return None


def lower_casing(df: pd.DataFrame, review_column: str ='review') -> pd.DataFrame:
    """
    Writes everything in lower case.
    
    Args:
        df: pd.DataFrame -> Dataframe containing the reviews.
        review_column: str = 'review' -> Name of the column containing the reviews.    
    
    Returns
        pd.Dataframe -> Dataset with lower cases only.
    """
    try:
        df[review_column] = df[review_column].str.lower()
        logger.info('Removed any upper case')
        return df
    
    except Exception as e:
        logger.error(f'Error while converting everything to lower case: {e}')
        return None
    
    

def apply_preprocessing(df: pd.DataFrame, review_column: str ='review'):
    """
    Applies preprocessing of the reviews and saves the results to a new csv-file.
    
    Args:
        df: pd.DataFrame -> Dataframe containing the reviews.
        review_column: str = 'review' -> Name of the column containing the reviews.
    """
    try:
        df = remove_br_tags(df=df, review_column='review')
        df = lower_casing(df=df, review_column='review')
        df = apply_treebank_tokenization(df=df)
        df = apply_stopword_removal(df=df)
        
        threads = [
            threading.Thread(target=apply_lemmatization, args=(df, 'tokenized', 'lemmatized')),
            threading.Thread(target=apply_stemming, args=(df, 'tokenized', 'stemmed')),
            threading.Thread(target=apply_lemmatization, args=(df, 'no_stopwords', 'lemmatized_no_sw')),
            threading.Thread(target=apply_stemming, args=(df, 'no_stopwords', 'stemmed_no_sw')),
            threading.Thread(target=positive_negative_converter, args=(df, 'sentiment'))
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        df.to_csv('data/preprocessed/preprocessed_dataset.csv', index=False)
        df.to_pickle('data/preprocessed/preprocessed_dataset.pkl')
        logger.info('Preprocessing completed. Data stored.')
        return df
    except Exception as e:
        logger.error(f'Error while preprocessing: {e}')


if __name__=='__main__':
    print('Start preprocessing the dataset.')
    check_and_download_nltk_resources()
    
    df = csv_reader('data/raw/imdb_dataset.csv')
    df_preprocessed = apply_preprocessing(df=df)
    print('Preprocessing finished:')
    print(df_preprocessed.info())
    