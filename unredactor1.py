
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import warnings
import os
import joblib
warnings.filterwarnings("ignore")


# Function to extract TF-IDF embeddings
def get_tfidf_embeddings(texts):
    """
    Generate TF-IDF embeddings for a list of texts.
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=300)
    tfidf_embeddings = tfidf_vectorizer.fit_transform(texts).toarray()
    return tfidf_vectorizer, tfidf_embeddings

# Updated Feature Extraction
def extract_features_with_sentiment(df, tfidf_vectorizer, dict_vectorizer=None):
    """
    Extract linguistic, semantic, positional, and statistical features.
    """
    feature_dicts = []
    redacted_texts = df['redacted_text'].tolist()

    tfidf_embeddings = tfidf_vectorizer.transform(redacted_texts).toarray()

    for index, row in df.iterrows():
        redacted_text = row['redacted_text']
        doc = nlp(redacted_text)

        # Basic Features
        num_tokens = len(doc)
        num_redactions = redacted_text.count('█')

        # Sentiment Features
        sentiment_scores = sia.polarity_scores(redacted_text)
        sentiment_neg = sentiment_scores['neg']
        sentiment_neu = sentiment_scores['neu']
        sentiment_pos = sentiment_scores['pos']

        # Contextual Features
        block_tokens = [token for token in doc if '█' in token.text]
        block_token_index = block_tokens[0].i if block_tokens else 0
        next_token = doc[block_token_index + 1] if block_token_index + 1 < len(doc) else ''
        prev_token = doc[block_token_index - 1] if block_token_index - 1 >= 0 else ''
        next_pos = next_token.pos_ if next_token else ''
        prev_pos = prev_token.pos_ if prev_token else ''

        # Positional Features
        redaction_position = (redacted_text.find('█') + 1) / len(redacted_text) if '█' in redacted_text else 0

        # Statistical Features
        sentence_lengths = [len(sent) for sent in nltk.sent_tokenize(redacted_text)]
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0

        # Semantic Features
        left_context_vector = np.mean([token.vector for token in doc[max(0, block_token_index - 3):block_token_index]], axis=0)
        right_context_vector = np.mean([token.vector for token in doc[block_token_index + 1:block_token_index + 4]], axis=0)

        dependency_rel = [token.dep_ for token in doc[max(0, block_token_index - 3):block_token_index]]


        # Create Feature Dictionary
        feature_dict = {
            'num_tokens': num_tokens,
            'num_redactions': num_redactions,
            'sentiment_neg': sentiment_neg,
            'sentiment_neu': sentiment_neu,
            'sentiment_pos': sentiment_pos,
            'redaction_position': redaction_position,
            'avg_sentence_length': avg_sentence_length,
            'next_token_text': next_token.text if next_token else '',
            'next_token_pos': next_pos,
            'prev_token_text': prev_token.text if prev_token else '',
            'prev_token_pos': prev_pos,
            'left_context_vector_sum': np.sum(left_context_vector),
            'right_context_vector_sum': np.sum(right_context_vector),
            'dependency_rel': dependency_rel
        }

        current_index = df.index.get_loc(index)  # Get the row's index in the current df

        # Add TF-IDF features
        for i, value in enumerate(tfidf_embeddings[current_index]):
            feature_dict[f'tfidf_feature_{i}'] = value

        feature_dicts.append(feature_dict)

    # Vectorize features
    if dict_vectorizer is None:
        dict_vectorizer = DictVectorizer(sparse=False)
        vectorized_features = dict_vectorizer.fit_transform(feature_dicts)
    else:
        vectorized_features = dict_vectorizer.transform(feature_dicts)

    return feature_dicts, dict_vectorizer, vectorized_features


def split_data(df):
    trainig_data = df[df['training_validation'] == 'training']
    validation_data = df[df['training_validation'] == 'validation']

    return trainig_data, validation_data

def train_model(vectorized_features_train, training_df):
    
    # Initialize Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=200,      # Number of trees
        max_depth=10,          # Maximum depth of the tree
        min_samples_split=2,   # Minimum number of samples required to split an internal node
        min_samples_leaf=2,    # Minimum number of samples required to be at a leaf node
        random_state=42        # For reproducibility
    )

    # Train the model
    rf_model.fit(vectorized_features_train, training_df['names'].tolist())

    
    
    return rf_model


def evaluate_model(rf_model, validation_df, vectorized_features_val):
    
    # Predict on validation data
    y_pred = rf_model.predict(vectorized_features_val)

    # print(set(y_pred))

    # Calculate evaluation metrics
    accuracy = accuracy_score(validation_df['names'].tolist(), y_pred)
    precision = precision_score(validation_df['names'].tolist(), y_pred, average='macro')
    recall = recall_score(validation_df['names'].tolist(), y_pred, average='macro')
    f1 = f1_score(validation_df['names'].tolist(), y_pred, average='macro')

    # Create a metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    # Print metrics
    print("Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    # print("\nClassification Report:")
    # print(classification_report(validation_df['names'].tolist(), y_pred))
    return y_pred


def dump_predicitons_to_file(rf_model, vectorized_features_val):
    y_pred = rf_model.predict(vectorized_features_val)

    with open('submission.tsv', 'w') as f:
        for idx, name in enumerate(y_pred):
            f.write(str(idx+1)+"\t"+name+'\n')


if __name__ == '__main__':

    # Download VADER lexicon (if not already downloaded)
    nltk.download('vader_lexicon')

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Load the dataset
    df = pd.read_csv('resources/unredactor.tsv', sep='\t', on_bad_lines='skip', names=['training_validation','names','redacted_text'])

    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    training_df, validation_df = split_data(df)

    all_texts = pd.concat([training_df['redacted_text'], validation_df['redacted_text']])

    # Fit TF-IDF vectorizer on all text data
    tfidf_vectorizer = TfidfVectorizer(max_features=300)
    tfidf_vectorizer.fit(all_texts)

    # Extract TF-IDF embeddings
    # tfidf_vectorizer, tfidf_embeddings = get_tfidf_embeddings(df['redacted_text'].tolist())

    feature_dicts, dict_vectorizer, vectorized_features = extract_features_with_sentiment(training_df, tfidf_vectorizer)
    if  not validation_df.empty:
        # The dict_vectorizer fit on the training data is used to transform the validation data, to ensure consistent features
        feature_dicts_val, _, vectorized_features_val = extract_features_with_sentiment(validation_df, tfidf_vectorizer, dict_vectorizer=dict_vectorizer)

    if os.path.exists('resources/training_model.pkl'):
        model = joblib.load('resources/training_model.pkl')
    else:
        # Train the model
        model = train_model(vectorized_features, training_df)
        joblib.dump(model, 'resources/training_model.pkl')

    # Evaluate the model
    print("Training Evaluation:")
    evaluate_model(model, training_df, vectorized_features)
    
    if  not validation_df.empty:
        print("Validation Evaluation:")
        evaluate_model(model, validation_df, vectorized_features_val)

    #-------------------
    # ---------------------------------------
    filename = sys.argv[1]
    
    if not filename:
        print("Please provide a filename to unredact.")
        sys.exit(1)
    
    df1 = pd.read_csv(filename, sep='\t', on_bad_lines='skip', names=['serial_number','redacted_text'])
    
    tfidf_vectorizer1 = TfidfVectorizer(max_features=300)
    tfidf_vectorizer1.fit(pd.concat([df['redacted_text'], df1['redacted_text']]))

    total_feature_dicts, total_dict_vectorizer, total_vectorized_features = extract_features_with_sentiment(df, tfidf_vectorizer1)

    test_feature_dicts, test_dict_vectorizer, test_vectorized_features = extract_features_with_sentiment(df1, tfidf_vectorizer1, dict_vectorizer=total_dict_vectorizer)

    if os.path.exists('resources/testing_model.pkl'):
        model1 = joblib.load('resources/testing_model.pkl')
    else:
        model1 = train_model(total_vectorized_features, df)
        joblib.dump(model1, 'resources/testing_model.pkl')

    # evaluate_model(model1, df1, test_vectorized_features)
    dump_predicitons_to_file(model1, test_vectorized_features)