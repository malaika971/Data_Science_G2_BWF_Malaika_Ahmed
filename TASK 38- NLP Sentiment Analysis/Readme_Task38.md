# **Task 38 NLP (SENTIMENT ANALYSIS)**
## Objective
This project focuses on building a Sentiment Analysis model to classify movie reviews as positive, negative, or neutral. The task involves data preprocessing, training machine learning models, and evaluating their performance.
## Dataset
The dataset used is the IMDB dataset of 50,000 movie reviews (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
## Libraries Used
```python
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import warnings
warnings.filterwarnings('ignore')
```

## Data Preprocessing
The following preprocessing steps were applied to the text data:
1. **HTML tag removal** using regex.
2. **Lowercasing** the text.
3. **URL removal**.
4. **Punctuation removal**.
5. **Stopword removal** using NLTK's stopwords corpus.
6. **Tokenization and Lemmatization** to standardize words.

## Exploratory Data Analysis (EDA)
- The dataset contains 50,000 rows, with two columns: `review` and `sentiment`.
- The sentiment distribution is nearly balanced, with `positive` and `negative` reviews almost equal.

## Model Training
- **Label Encoding**: The sentiment labels (`positive`, `negative`) were encoded into numerical values.
- **Train-Test Split**: 75% of the data was used for training, and 25% for testing.
- **Vectorization**: Used **CountVectorizer** to convert text data into numerical features.

### Models Used
1. **Logistic Regression**: A linear model used for classification tasks.
2. **Multinomial Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
3. **Bernoulli Naive Bayes**: Similar to MultinomialNB but assumes binary features.

### Model Evaluation
The accuracy scores for each model were evaluated on the test set:
- **Logistic Regression**: 88.23%
- **Multinomial Naive Bayes**: 85.64%
- **Bernoulli Naive Bayes**: 85.57%

A bar chart comparing these accuracy scores is included.



![MA](https://github.com/user-attachments/assets/5b2ad9f0-e3e6-44fb-8461-54418a523b42)


## User Input Prediction
  - The trained models were used to predict the sentiment of user input:
  - For input **"I love this movie!"**, the prediction was **Positive**.
  - For input **"Hating the movie"**, the prediction was **Negative**.

## Saving the Models
The trained models and vectorizer were saved using **joblib** for future use:
- `lr_model.pkl` for Logistic Regression
- `mn_model.pkl` for Multinomial Naive Bayes
- `bn_model.pkl` for Bernoulli Naive Bayes
- `vectorizer.pkl` for the fitted CountVectorizer


## Conclusion
This project demonstrates how to build a sentiment analysis model from scratch, applying text preprocessing techniques, training machine learning models, and evaluating their performance The models can be used to classify movie reviews or other text data based on sentiment.
