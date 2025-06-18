# Customer-Sentiment-Analysis-in-Restaurant-Reviews-Using-NLP-


Project Overview

This project analyzes customer sentiments in restaurant reviews using Natural Language Processing (NLP) techniques. The goal is to classify customer feedback as positive or negative, helping restaurants understand customer satisfaction and improve their services.

Features
Text preprocessing: tokenization, stopword removal, stemming

Feature extraction using Bag of Words (BoW)

Machine learning classification models (Logistic Regression, Random Forest, etc.)

Model evaluation with accuracy, confusion matrix, and classification reports

Dataset
The dataset contains restaurant reviews labeled as positive or negative. It includes:

Review text

Sentiment label (positive/negative)

Installation
Clone the repository:

git clone https://github.com/yourusername/Customer-Sentiment-Analysis-in-Restaurant-Reviews-Using-NLP.git

cd Customer-Sentiment-Analysis-in-Restaurant-Reviews-Using-NLP

Install required packages:

pip install -r requirements.txt

Download necessary NLTK data:

python

import nltk

nltk.download('stopwords')

nltk.download('punkt')

Usage

Open and run the Jupyter Notebook Customer_Sentiment_Analysis.ipynb to see the full data processing, model training, and evaluation steps.

Model Details

Used Logistic Regression and Random Forest classifiers

Preprocessing steps: cleaning, tokenization, stopword removal, stemming

Vectorized text data using Bag of Words technique

Achieved accuracy of approximately XX% (replace with your result)

Folder Structure
bash

Copy

Edit

├── Customer_Sentiment_Analysis.ipynb   # Jupyter notebook with full code

├── requirements.txt                   # Required Python packages    

│   └── restaurant_reviews.csv

└── README.md                         # Project documentation

Future Work

Experiment with advanced text representations like TF-IDF, Word2Vec, or transformers

Implement hyperparameter tuning with GridSearchCV for improved accuracy

Expand dataset with more reviews for better generalization

Develop and deploy a web app for easy sentiment prediction

Author

Rutika Sahane
Email: rutikasahane67@gmail.com


