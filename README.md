# Movie Review Sentiment Analysis

## Table of Content

1. [**Background Information**](#background-information)
2. [**Methodology**](#methodology)
3. [**Data and Information Extraction**](#data-and-information-extraction)
4. [**Data Preprocessing**](#data-preprocessing)
5. [**Model Training**](#model-training)
6. [**Model Evaluation**](#model-evaluation)
7. [**Classification Application**](#classification-application)
8. [**Using This Repository**](#using-this-repository)

# **Background Information**


# **Methodology**


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

[Code](src\preprocessing\read_data.py)  
[Data](data)  


# **Data Preprocessing**

## Preprocessing
There are four typical preprocessing tasks.  
- Tokenization
- Stop Word Removal
- Lemmatization
- Stemming

All are applied to the dataset:  
The lemmatization and the stemming are applied before and after the stop word removal.  
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

The vectorization results are summarized (generally based on average) for each review, so that each review has one vector per tokenized column.  
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
- Decision Tree
- Fully Connected Artificial Neural Network

## ML Model Training
Considering the preprocessing and vectorization steps in total 54 models are possible (6 preprocessing steps x 3 vectorization methods x 3 ML approaches).  

In the first round a small portion of the dataset was used to train the model (10%; 5,000 reviews).  
In this round the hyperparameters for each model were fine-tuned.

For the second round only the best performing approaches are selected:
- The four best preprocessing methods.
- The two best vectorization methods.  
- The two best ML approaches.
In this round 30% of the data was used for training.  
All remaining models were fine-tuned.  

For the third round the best approach was selected and fine-tuned using 80% of the dataset for training.

# **Model Evaluation**


# **Classification Application**


# **Using This Repository**

