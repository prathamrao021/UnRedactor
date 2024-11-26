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


# Function to extract TF-IDF embeddings
def get_tfidf_embeddings(texts):
    """
    Generate TF-IDF embeddings for a list of texts.
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=300)
    tfidf_embeddings = tfidf_vectorizer.fit_transform(texts).toarray()
    return tfidf_vectorizer, tfidf_embeddings



# Function to extract features and include sentiment scores
def extract_features_with_sentiment(df, tfidf_vectorizer, dict_vectorizer=None):
    """
    Extract linguistic features, sentiment (neu, neg, pos), and TF-IDF embeddings.
    """
    feature_dicts = []
    redacted_texts = df['redacted_text'].tolist()

    # Get TF-IDF embeddings
    tfidf_embeddings = tfidf_vectorizer.transform(redacted_texts).toarray()

    for index, row in df.iterrows():
        redacted_text = row['redacted_text']
        doc = nlp(redacted_text)

        # Number of tokens and redactions
        num_tokens = len(doc)
        num_redactions = redacted_text.count('█')

        # Sentiment analysis using VADER
        sentiment_scores = sia.polarity_scores(redacted_text)
        sentiment_neg = sentiment_scores['neg']
        sentiment_neu = sentiment_scores['neu']
        sentiment_pos = sentiment_scores['pos']

        # Previous and next tokens
        block_tokens = [token for token in doc if '█' in token.text]
        if not block_tokens:
            next_token = ''
            prev_token = ''
        else:
            block_token = block_tokens[0]
            block_token_index = block_token.i

            next_token_index = min(block_token_index + num_redactions, len(doc) - 1)
            next_token = doc[next_token_index] if next_token_index < len(doc) else ''

            prev_token_index = max(0, block_token_index - 1)
            prev_token = doc[prev_token_index] if prev_token_index >= 0 else ''

        next_pos = next_token.pos_ if next_token else ''
        prev_pos = prev_token.pos_ if prev_token else ''

        left_context_text = " ".join([token.text for token in doc[block_token_index : min(block_token_index + 3, len(doc))]])
        right_context_text = " ".join([token.text for token in doc[max(0, block_token_index - 3) : block_token_index]])

        left_tfidf = tfidf_vectorizer.transform([left_context_text]).toarray()[0] if left_context_text else np.zeros(tfidf_vectorizer.max_features) # Handle empty context
        right_tfidf = tfidf_vectorizer.transform([right_context_text]).toarray()[0] if right_context_text else np.zeros(tfidf_vectorizer.max_features) # Handle empty context

        # Create feature dictionary
        feature_dict = {
            'num_tokens': num_tokens,
            'num_redactions': num_redactions,
            'sentiment_neg': sentiment_neg,
            'sentiment_neu': sentiment_neu,
            'sentiment_pos': sentiment_pos,
            'next_token_text': next_token.text if next_token else '',
            'next_token_pos': next_pos,
            'prev_token_text': prev_token.text if prev_token else '',
            'prev_token_pos': prev_pos,
            'left_context_tfidf': left_tfidf.sum(),
            'right_context_tfidf': right_tfidf.sum(),
        }

        current_index = df.index.get_loc(index)  # Get the row's index in the current df

        # Add TF-IDF features
        for i, value in enumerate(tfidf_embeddings[current_index]):
            feature_dict[f'tfidf_feature_{i}'] = value

        feature_dicts.append(feature_dict)

    # Vectorize feature dictionaries
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

def train_model(vectorized_features_train, training_df, vectorized_features_val, validation_df):
    
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

    # print(y_pred)

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
    print("Validation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    # print("\nClassification Report:")
    # print(classification_report(validation_df['names'].tolist(), y_pred))

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

    # The dict_vectorizer fit on the training data is used to transform the validation data, to ensure consistent features
    feature_dicts_val, _, vectorized_features_val = extract_features_with_sentiment(validation_df, tfidf_vectorizer, dict_vectorizer=dict_vectorizer)

    # Train the model
    model = train_model(vectorized_features, training_df, vectorized_features_val, validation_df)

    # Evaluate the model
    evaluate_model(model, validation_df, vectorized_features_val)