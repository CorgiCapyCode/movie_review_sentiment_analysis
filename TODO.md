2025-03-21:  
- Implement code for self-training of Word2Vec, RoBERTa and ELMo
- Conduct the vectorization 
    - store the weights
    - store the results (csv) -> due to size of the results store in different files per vecotirzation method
    - vectorization must be for each column (e.g. only lemmatization, only stemming, stemming with stop word removal)


2025-03-22:
- Implement the training algorithms
    - select three different methods
    - apply the methods on each of the vectorized columns
    - conduct training for different sample sizes
        - split into training, test and validation datasets
        - create samples for 1% (500), 5% (2500), 10% (5000), 20% (10000), 25% (12500), 40% (20000), 50% (25000), 75% (37500), 100% (50000)
        - consider that the data is split!

- Write code for evaluation
    - generate overview about the classification (TP, FP, TN, FN)
    - calculate the standard metrics (i.e. F-score, accuracy, precision, recall)
    - calculate (if possible) ROC and AUC

- Compare the results and select the best algorithm

- Create a simple container app, that reads in the review and outputs whether the review is good or bad



Data Structure:

Pre-trained:  

review:             The original review  
sentiment:          The binary sentiment  
tokenized:          The original tokenized column  
no_stopwords:       Stopwords removed  
stemmed_no_sw:      Stemming with stop words removed  
stemmed:            stemmed, but stop words included  
lemmatized:         lemmatization, but with stop words included  
lemmatized_no_sw:   lemmatization with stop words removed  

- Vectorization:
    - six vectorization results per algorithm
    - apply on tokenized, no_stopwords, stemmed_no_sw, stemmed, lemmatized, lemmatized_no_sw

    - probably best approach to create single vectorized file for each algorithm
    - check if replacement of words is recommended (not for the first column, in order to maintain one text column)




https://medium.com/@abhishekrajs121098/complete-guide-for-supervised-learning-classification-machine-learning-a7ea4e9ff676


Find a solution to store the large datasets! 





1. Implement SVM
2. Implement Decision Trees
3. Own ANN
