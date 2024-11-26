import pytest
import pandas as pd
import spacy
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.sentiment import SentimentIntensityAnalyzer
from unredactor import extract_features_with_sentiment, evaluate_model, dump_predicitons_to_file, train_model
import os
import joblib

def test_extract_features_with_sentiment():
    nlp = spacy.load('en_core_web_sm')
    sia = SentimentIntensityAnalyzer()
    df = pd.read_csv("resources/unredactor.tsv", sep='\t', on_bad_lines='skip', names=['training_validation','names','redacted_text'])
    tfidf_vectorizer = TfidfVectorizer(max_features=300)
    tfidf_vectorizer.fit(df['redacted_text'])
    feature_dicts, dict_vectorizer, vectorized_features = extract_features_with_sentiment(sia, nlp, df, tfidf_vectorizer)
    assert vectorized_features is not None

def test_train_model():
    nlp = spacy.load('en_core_web_sm')
    sia = SentimentIntensityAnalyzer()
    df = pd.read_csv("resources/unredactor.tsv", sep='\t', on_bad_lines='skip', names=['training_validation','names','redacted_text'])
    tfidf_vectorizer = TfidfVectorizer(max_features=300)
    tfidf_vectorizer.fit(df['redacted_text'])
    feature_dicts, dict_vectorizer, vectorized_features = extract_features_with_sentiment(sia, nlp, df, tfidf_vectorizer)
    # Train the model
    model = train_model(vectorized_features, df)
    assert model is not None

def test_evaluate_model():
    nlp = spacy.load('en_core_web_sm')
    sia = SentimentIntensityAnalyzer()
    df = pd.read_csv("resources/unredactor.tsv", sep='\t', on_bad_lines='skip', names=['training_validation','names','redacted_text'])
    tfidf_vectorizer = TfidfVectorizer(max_features=300)
    tfidf_vectorizer.fit(df['redacted_text'])
    feature_dicts, dict_vectorizer, vectorized_features = extract_features_with_sentiment(sia, nlp, df, tfidf_vectorizer)
    # Train the model
    model = train_model(vectorized_features, df)
    joblib.dump(model, 'resources/training_model.pkl')
    y = evaluate_model(model, df, vectorized_features)
    assert y[0]
