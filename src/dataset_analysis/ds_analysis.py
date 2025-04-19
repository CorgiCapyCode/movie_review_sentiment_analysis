import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
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


def convert_json_to_csv(data_path: str):
    """
    Converts a JSON file to a CSV file.

    Args:
        data_path: str -> Path to the JSON file.
    """
    try:
        with open (data_path, 'r') as json_file:
            data = json.load(json_file)
            
        test_results = data['test_results']
        rows = []
        for model, datasets in test_results.items():
            for dataset, metrics in datasets.items():
                row = {
                    'model': model,
                    'dataset': dataset,
                    'accuracy': metrics['accuracy'],
                    'f1_score': metrics['f1_score'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall']
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(f'{data_path.replace(".json", ".csv")}', index=False)
    except Exception as e:
        print(f"Error converting JSON to CSV: {e}")


def plot_accuracy_histogram(data: pd.DataFrame, accuracy_column: str = 'accuracy', model_column: str = 'model', bin_width: float = 0.02, save_path: str = 'accuracy_histogram.png'):
    """
    Plots a histogram of accuracy results clustered in 2% steps, with bars divided by model type.

    Args:
        data: pd.DataFrame -> The input data containing accuracy and model columns.
        accuracy_column: str -> The name of the column containing accuracy values.
        model_column: str -> The name of the column containing classifiers.
        bin_width: float (0.02) -> The width of each accuracy bin
        save_path: str -> The file path to save the histogram image.
    """
    try:
        min_accuracy = np.floor(data[accuracy_column].min() / bin_width) * bin_width
        max_accuracy = np.ceil(data[accuracy_column].max() / bin_width) * bin_width
        bins = np.arange(min_accuracy, max_accuracy + bin_width, bin_width)

        colors = {
            'ANN': '#90ee90',  # Light green
            'SVM': '#1f77b4',  # Dark blue
            'RF': '#ff7f0e'    # Orange
        }

        model_histograms = {model: np.zeros(len(bins) - 1) for model in colors.keys()}

        for model in colors.keys():
            model_data = data[data[model_column] == model]
            model_histograms[model], _ = np.histogram(model_data[accuracy_column], bins=bins)

        bottom = np.zeros(len(bins) - 1)
        plt.figure(figsize=(10, 6))
        for model, color in colors.items():
            plt.bar(bins[:-1], model_histograms[model], width=bin_width, color=color, edgecolor='black', alpha=0.8, label=model, bottom=bottom)
            bottom += model_histograms[model]

        plt.xlabel('Accuracy Range')
        plt.ylabel('Number of Results')
        plt.title('Histogram of Accuracy Results by Model Type')

        plt.xticks(bins[:-1] + bin_width / 2, [f'{int(b*100)}-{int((b+bin_width)*100)}%' for b in bins[:-1]], rotation=45)

        plt.legend(title='Model Type')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Histogram saved to {save_path}")
    except Exception as e:
        print(f"Error plotting accuracy histogram: {e}")



if __name__ == '__main__':
    
    analysis = False
    see_results = True
    phase_bar_chart = False
    
    if analysis:
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

    if see_results:
        convert_json_to_csv('data/training/phase_3/results_and_hyperparams_r3.json')
        print('Finished')

    
    if phase_bar_chart:
        df = csv_reader('data/training/phase_3/phase_3_all_results.csv')
        
        plot_accuracy_histogram(
            data=df,
            accuracy_column='accuracy',
            model_column='model',
            bin_width=0.01,
            save_path='data/training/phase_3/phase_3_accuracy_histogram.png'
        )