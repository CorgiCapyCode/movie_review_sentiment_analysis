import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing.read_data import csv_reader, pickle_reader

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def count_sentiments(df: pd.DataFrame) -> dict:
    """
    Counts the number of positive and negative sentiments in the dataset.

    Args:
        df: pd.DataFrame -> The dataset.

    Returns:
        dict: A dictionary with counts of positive and negative sentiments.
    """
    sentiment_counts = df['sentiment'].value_counts()
    return {
        'positive': sentiment_counts.get('positive', 0),
        'negative': sentiment_counts.get('negative', 0)
    }


def calculate_stats(token_counts):
    """
    Calculates the longest, shortest, and average token counts.
    
    Args:
        token_counts: list -> A list of token counts for each review.
    """
    if not token_counts:
        return {'longest': 0, 'shortest': 0, 'average': 0}
    return {
        'longest_review': max(token_counts),
        'shortest_review': min(token_counts),
        'average_review': sum(token_counts) / len(token_counts)
    }


def analyze_review_lengths(df: pd.DataFrame, column_name: str) -> dict:
    """
    Analyzes the number of tokens in each review for a specified column.

    Args:
        df: pd.DataFrame -> The dataset.
        column_name: str -> The name of the column containing tokens per review.

    Returns:
        dict -> A dictionary with the token counts for each review, the longest review, and the shortest review.
    """
    if column_name not in df.columns:
        raise ValueError(f'Column {column_name} does not exist in the dataset.')
    
    all_review_token_counts = df[column_name].apply(lambda x: len(x) if isinstance(x, list) else 0).tolist()
    all_reviews_stats = calculate_stats(all_review_token_counts)
    
    token_counts_positive = df[df['sentiment'] == 'positive'][column_name].apply(lambda x: len(x) if isinstance(x, list) else 0).tolist()
    positive_stats = calculate_stats(token_counts_positive)

    token_counts_negative = df[df['sentiment'] == 'negative'][column_name].apply(lambda x: len(x) if isinstance(x, list) else 0).tolist()
    negative_stats = calculate_stats(token_counts_negative)
    
    return {
        'overall': {
            'token_counts': all_review_token_counts,
            **all_reviews_stats
        },
        'positive': {
            'token_counts': token_counts_positive,
            **positive_stats
        },
        'negative': {
            'token_counts': token_counts_negative,
            **negative_stats
        }
    }


def plot_token_histogram_stacked(overall_counts: list, positive_counts: list, negative_counts: list, save_path: str, title: str = 'Token Count Histogram', bins: int = 20):
    """
    Plots a stacked histogram of token counts, splitting each bar into positive and negative parts.

    Args:
        overall_counts: list -> A list of overall token counts for each review.
        positive_counts: list -> A list of token counts for positive reviews.
        negative_counts: list -> A list of token counts for negative reviews.
        title: str -> The title of the histogram.
        bins: int -> The number of bins for the histogram.
        save_path: str -> The path to save the histogram image.
    """
    if not overall_counts or not positive_counts or not negative_counts:
        raise ValueError('One or more token count lists are empty.')

    # Create the histogram bins
    bin_edges = plt.hist(overall_counts, bins=bins, alpha=0)[1]  # Get bin edges without plotting

    # Create histograms for positive and negative counts
    positive_hist, _ = np.histogram(positive_counts, bins=bin_edges)
    negative_hist, _ = np.histogram(negative_counts, bins=bin_edges)

    # Plot the stacked histogram
    plt.figure(figsize=(20, 6))
    plt.bar(bin_edges[:-1], positive_hist, width=np.diff(bin_edges), color='blue', edgecolor='black', alpha=0.7, label='Positive')
    plt.bar(bin_edges[:-1], negative_hist, width=np.diff(bin_edges), color='red', edgecolor='black', alpha=0.7, bottom=positive_hist, label='Negative')

    # Add labels and title
    plt.title(title)
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig(save_path)
    print(f"Histogram saved at {save_path}")



if __name__ == '__main__':
    df = pickle_reader('data/preprocessed/preprocessed_dataset.pkl')
    print(df.info())
    #print(df.head())
    #print(df.columns)

    sentiment_counts = count_sentiments(df)
    print('Sentiment counts:', sentiment_counts)
    
    
    
    feature_columns = [
    'tokenized',
    'no_stopwords',
    'stemmed_no_sw',
    'stemmed',
    'lemmatized',
    'lemmatized_no_sw'    
    ]
    
    
    for feature in feature_columns:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame columns.")
        token_analysis = analyze_review_lengths(df, feature)
        print(f"Token analysis for {feature}:")
        print(f"Longest review: {token_analysis['overall']['longest_review']}")
        print(f"Shortest review: {token_analysis['overall']['shortest_review']}")
        print(f"Average review length: {token_analysis['overall']['average_review']}")
        print(f"Longest positive review: {token_analysis['positive']['longest_review']}")
        print(f"Shortest positive review: {token_analysis['positive']['shortest_review']}")
        print(f"Average positive review length: {token_analysis['positive']['average_review']}")
        print(f"Longest negative review: {token_analysis['negative']['longest_review']}")
        print(f"Shortest negative review: {token_analysis['negative']['shortest_review']}")
        print(f"Average negative review length: {token_analysis['negative']['average_review']}")
    
    
    tokenized_results = analyze_review_lengths(df, 'tokenized')

    plot_token_histogram_stacked(
        overall_counts=tokenized_results['overall']['token_counts'],
        positive_counts=tokenized_results['positive']['token_counts'],
        negative_counts=tokenized_results['negative']['token_counts'],
        save_path='data/preprocessed/token_histogram_original.png',
        title='Token Count Histogram - Original',
        bins=100)
