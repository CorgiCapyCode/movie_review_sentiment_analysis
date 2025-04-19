# Movie Review Sentiment Analysis

## Requirements:
The following resources need to be downloaded separately:  
[GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g)  

please store in: [src\preprocessing\vec_model_weights](src\preprocessing\vec_model_weights)

Please run the preprocessing and vectorization scripts before training. Otherwise the input data is not available.  
The two dataset are to large to upload them direct to on GitHub.  

Thanks for your understanding.  

## Table of Content

1. [**Introduction**](#background-information)
2. [**Data and Information Extraction**](#data-and-information-extraction)
3. [**Data Preprocessing**](#data-preprocessing)
4. [**Model Training**](#model-training)
5. [**Model Evaluation**](#model-evaluation)
6. [**Classification Application**](#classification-application)
7. [**Using This Repository**](#using-this-repository)
8. [**Result Structure**](#result-structure)


# **Introduction**
This project is about sentiment classification of the IMDb Dataset of Movie Reviews.  
It contains modules for preprocessing the dataset, review vectorization and model training.  
Different approaches for each of the modules mentioned above were combined, trained and evaluated.  
The model weights of the best performing model are stored and can be downloaded separately. [Link to Download](data/training/models)  

# **Data and Information Extraction**

For this project the stanford dataset for movie reviews was used.  
[Stanford's Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  

```bibtex
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```

The original dataset consists of txt-files with a rating. The following preprocessed version was used:  
[IMDB Dataset of 50k Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  

The preprocessing consists of the following steps:  
- Summarizing the txt-files to one csv-file.
- Replacing the rating (1-10) with a binary classifier (positive/negative)

There is a [function](src\preprocessing\read_data.py) available to convert the txt-file library to a single csv-file.  
Please check the required file structure, described in the code.  

[Code](src\preprocessing\read_data.py)  
[Data](data)  


# **Data Preprocessing**

## Preprocessing
There are four typical preprocessing tasks.  
- Tokenization
- Stop-Word Removal
- Lemmatization
- Stemming

All are applied to the dataset:  
The lemmatization and the stemming are applied before and after the stop-word removal.  
For lemmatizaiton POS-tagging is applied to enhance the results.  

The preprocessing results in six different tokenized columns:  
- tokenized
- no_stopwords
- stemmed
- lemmatized
- stemmed_no_sw
- lemmatized_no_sw

The results are stored as PKL-file as well as CSV-file:  
[Code](src\preprocessing\preprocessing.py)  
[Data](data\preprocessed)   

## Vectorization
Vectorization is applied using different methods:  
- Word2Vec
- RoBERTa

For Word2Vec a pre-trained model was used: [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g)  
The model weights for the pre-trained algorithm are excluded from this repo. Please download the latest weights from the link above.

Furthermore a self-trained version of Word2Vec is available: [Model weights for self-trained Word2Vec](src\preprocessing\vec_model_weights)  
The model weights for RoBERTa are loaded from the transformers library.  
No additional (transfer) training was conducted on the pre-trained models.  

The vectorization results are aggregated (based on average) for each review, so that each review has one vector per tokenized column.  
The weights are stored [here](src\preprocessing\vec_model_weights).  
The results are stored separately for each vectorization method as PKL as well as as CSV-file.  

[Code](src\preprocessing\vectorization.py)  
[Data](data\vectorized)

# **Model Training**
## Description of Input Data
There are three different datasets available. One for each vectorization method.  
- Word2Vec based on GoogleNews
- Self-trained Word2Vec
- pre-trained RoBERTa

Each of the datasets contains six different columns with vectorized reviews. This results in 18 different inputs.  

## ML Model Selection
Three different models where chosen for the sentiment analysis:
- Support Vector Machine
- Random Forest
- Fully Connected Artificial Neural Network

## ML Model Training
Considering the preprocessing and vectorization steps in total 54 models are possible (6 preprocessing steps x 3 vectorization methods x 3 ML approaches).  

In the first phase a small portion of the dataset was used to train the model (10%; 5,000 reviews).  
In this phase the hyperparameters for each model were fine-tuned in three rounds.  

For the second phase only the best performing approaches are selected:
- The two best preprocessing methods.
- The two best vectorization methods.  
- The two best ML approaches.
In this phase 30% of the data was used for training.  
All remaining models were fine-tuned.  

For the third phase the best approach was selected and fine-tuned using 80% of the dataset for training.  


# **Model Evaluation**
The models are evaluated based on the following metrics:  
- Accuracy
- F1-Score
- Recall
- Precision

The selection process followed these steps:
1. Rank all results of a phase from highest to lowest accuracy.  
2. Select the best two methods (first appearing ones) for the three components: preprocessing, vectorization and classifier.

The results contain the confusion matrix as well.  
Note: The test-size of the dataset is always the remaining data. That means for the first round 90% of the dataset was used to test, while in the last round only 20% was used.  

# **Using This Repository**
The pipeline consists of the following steps, which can be executed individually:
- [Preprocessing](src\preprocessing\preprocessing.py)
- [Vectorization](src\preprocessing\vectorization.py)
- [Training & Testing](src\training\training.py)

To run all of the process a main is also available:
- [main.py](main.py)

It is recommended to run the training and testing process alone (not via main.py), since the preprocessing and vectorization results should not change.  
Running the main.py for every iteration, increases the computational costs. 

Please check the default paths used and ensure that the required data is available.  
**Minimum data required:**  
- imdb_dataset.csv in data\raw
- GoogleNews-vectors-negative300.bin in src\preprocessing\vec_model_weights

Then follow the pipeline (recommended) or run main.py. 

# **Result Structure**
When all data is stored in the default locations and the experiments have been conducted the results of each step should be stored as follows:  


|            |            |            |            |Description |Input for...|
|------------|------------|------------|------------|------------|------------|
|data|
| | -- preprocessed | | | Contains the results of the preprocessing steps. | | 
| | | -- preprocessed_dataset.csv / .pkl | | Preprocessed dataset as described in Preprocessing. | Used for vectorization. | 
| | | -- token_histogram_original.png | | Visualization of the token length distribution. | | 
| | -- raw | | | Contains the raw data. | | 
| | | -- imdb_dataset.csv | | Dataset downloaded from Kaggle. | Used for preprocessing. | 
| | -- training | | | Contains the results of the training process. | | 
| | | -- models | | Contains the model weights. | |
| | | | -- roberta_review_vector_tokenized_ann.keras | The final results. Model: Preprocessing: Tokenized, Vectorization: Preprocessed RoBERTa, Classifier: ANN | |
| | | -- phase_x | | Contains the documentation of phase x in three rounds. | |
| | | | -- phase_x_accuracy_histogram.png | Visualization of model accuracies over the three rounds of the phase. | |
| | | | -- phase_x_all_results.csv | Aggregation of all results in one file. | |
| | | | -- results_and_hyperparameters_ry.csv / .json / .pkl | The results of the round for each trained model, includes the training setup (hyperparameters and random state) | |
| | -- vectorized | | | Contains the results of the vectorization process. | |
| | | -- approach.csv / .pkl | | Results of the vectorization per method. | Used for training. | 