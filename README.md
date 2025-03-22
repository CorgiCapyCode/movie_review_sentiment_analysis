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

[Data](data)  
[Code](src\preprocessing\read_data.py)  

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

[Code](src\preprocessing\preprocessing.py)  

## Vectorization
Vectorization is applied using different methods:  
- Word2Vec
- RoBERTa

For Word2Vec a pre-trained model was used: [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g)  
The model weights for the pre-trained algorithm are excluded from this repo. Please download the latest weights 

[Code](src\preprocessing\vectorization.py)  



# **Model Training**


# **Model Evaluation**


# **Classification Application**


# **Using This Repository**

