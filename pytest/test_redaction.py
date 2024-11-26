import pytest
import pandas as pd
import spacy
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.sentiment import SentimentIntensityAnalyzer
from unredactor1 import extract_features_with_sentiment, evaluate_model, dump_predicitons_to_file


def test_extracted_feature():
    nlp = spacy.load('en_core_web_sm')
    sia = SentimentIntensityAnalyzer()
    df = pd.read_csv("resources/test1.txt", sep='\t', on_bad_lines='skip', names=['serial_number','redacted_text'])
    tfidf_vectorizer, tfidf_embeddings = extract_features_with_sentiment(df, nlp, sia)
    assert tfidf_embeddings

def test_evaluate_model():
    nlp = spacy.load('en_core_web_sm')
    sia = SentimentIntensityAnalyzer()
    df = pd.read_csv("resources/test1.txt", sep='\t', on_bad_lines='skip', names=['serial_number','redacted_text'])
    tfidf_vectorizer, tfidf_embeddings = extract_features_with_sentiment(df, nlp, sia)
    model = RandomForestClassifier()
    evaluate_model(model, df, tfidf_embeddings)