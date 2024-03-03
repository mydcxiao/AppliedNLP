'''
Python Version: 3.11.4
'''
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from bs4 import BeautifulSoup
import contractions
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

 
# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz

# Statistics Printing

def summary(y_train, pred_train, y_test, pred_test):
    print(f'Train accuracy: {accuracy_score(y_train, pred_train)},'
          f'Train precision: {precision_score(y_train, pred_train)},'
          f'Train recall: {recall_score(y_train, pred_train)},'
          f'Train f1 score: {f1_score(y_train, pred_train)},'
          f'Test accuracy: {accuracy_score(y_test, pred_test)},'
          f'Test precision: {precision_score(y_test, pred_test)},'
          f'Test recall: {recall_score(y_test, pred_test)},'
          f'Test f1 score: {f1_score(y_test, pred_test)}'
        )

# Read Data

dtype={7: object}
# url = 'https://web.archive.org/web/20201127142707if_/https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Office_Products_v1_00.tsv.gz'
# data = pd.read_csv(url, sep='\t', on_bad_lines='skip', dtype=dtype)
# path = 'amazon_reviews_us_Office_Products_v1_00.tsv.gz'
# data = pd.read_csv(path, sep='\t', on_bad_lines='skip', dtype=dtype)
unzipped_path = 'amazon_reviews_us_Office_Products_v1_00.tsv'
data = pd.read_csv(unzipped_path, sep='\t', on_bad_lines='skip', dtype=dtype)


# Keep Reviews and Ratings

data = data[['star_rating', 'review_headline', 'review_body']]
data.dropna(inplace=True)
data['star_rating'] = pd.to_numeric(data['star_rating'], errors='coerce')


#  We form three classes and select 100000 reviews randomly from positive and negtive class.

pos_reviews = data[data['star_rating'] > 3]
neg_reviews = data[data['star_rating'] <= 2]
neu_reviews = data[data['star_rating'] == 3]

print(f"Positive reviews: {pos_reviews.shape[0]}, Negative reviews: {neg_reviews.shape[0]}, Neutral reviews: {neu_reviews.shape[0]}")

pos_samples = pos_reviews.sample(n=100000, random_state=0)
neg_samples = neg_reviews.sample(n=100000, random_state=0)
pos_samples['star_rating'] = 1
neg_samples['star_rating'] = 0
pos_samples.rename(columns={'star_rating': 'label'}, inplace=True)
neg_samples.rename(columns={'star_rating': 'label'}, inplace=True)
dataset = pd.concat([pos_samples, neg_samples])
dataset['review'] = dataset[['review_headline', 'review_body']].agg(' '.join, axis=1)
dataset.drop(columns=['review_headline', 'review_body'], inplace=True)


# Data Cleaning

avg_len_before_clean = dataset['review'].str.len().mean()

from bs4 import MarkupResemblesLocatorWarning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
dataset['review'] = dataset['review'].apply(str.lower)
dataset['review'] = dataset['review'].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', x))
dataset['review'] = dataset['review'].apply(lambda x: BeautifulSoup(x, "html.parser").text)
dataset['review'] = dataset['review'].apply(lambda x: re.sub(r'[^a-zA-Z]+', ' ', x))
dataset['review'] = dataset['review'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
dataset['review'] = dataset['review'].apply(lambda x: ' '.join([contractions.fix(word) for word in x.split()]))

avg_len_after_clean = dataset['review'].str.len().mean()

print(f"Average length of reviews before and after data cleaning: {avg_len_before_clean}, {avg_len_after_clean}")


# Pre-processing

avg_len_before_prep = dataset['review'].str.len().mean()

# remove the stop words 

stopwords = set(stopwords.words('english'))
dataset['review'] = dataset['review'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stopwords]))
dataset.sample(3,random_state=1)


# perform lemmatization  

lemmatizer = WordNetLemmatizer()
dataset['review'] = dataset['review'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

avg_len_after_prep = dataset['review'].str.len().mean()

print(f"Average length of reviews before and after data preprocessing: {avg_len_before_prep}, {avg_len_after_prep}")


# TF-IDF Feature Extraction

vectorizer = TfidfVectorizer(ngram_range=(1, 3))
vectors = vectorizer.fit_transform(dataset['review'])


# Train-Test Split

labels = dataset['label']
x_train, x_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, random_state=42, stratify=labels)


# Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
pred_train = perceptron.predict(x_train)
pred_test = perceptron.predict(x_test)
print("Perceptron: ")
summary(y_train, pred_train, y_test, pred_test)


# SVM

svm = LinearSVC(dual='auto')
svm.fit(x_train, y_train)
pred_train = svm.predict(x_train)
pred_test = svm.predict(x_test)
print("SVM: ")
summary(y_train, pred_train, y_test, pred_test)


# Logistic Regression

logit = LogisticRegression(solver='saga', random_state=0)
logit.fit(x_train, y_train)
pred_train = logit.predict(x_train)
pred_test = logit.predict(x_test)
print("Logistic Regression: ")
summary(y_train, pred_train, y_test, pred_test)


# Naive Bayes

naive_bayes = MultinomialNB()
naive_bayes.fit(x_train, y_train)
pred_train = naive_bayes.predict(x_train)
pred_test = naive_bayes.predict(x_test)
print("Naive Bayes: ")
summary(y_train, pred_train, y_test, pred_test)

