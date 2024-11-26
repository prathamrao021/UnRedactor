# cis6930fa24-project2

## Overview

The Unredactor project aims to identify and unredact names from redacted text using various machine learning techniques. The project leverages natural language processing (NLP) tools, sentiment analysis and many other features to train a model to predict the redacted names.

## Features

- **TF-IDF Embeddings**: Extracts TF-IDF features from the redacted text.
- **Sentiment Analysis**: Uses VADER sentiment analysis to extract sentiment features.
- **Contextual Features**: Extracts features based on the context around the redacted tokens.
- **Positional Features**: Extracts features based on the position of the redacted tokens.
- **Statistical Features**: Extracts statistical features such as average sentence length.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/prathamrao021/cis6930fa24-project2
    cd cis6930fa24-project2
    ```

2. Install `pipenv` if you haven't already:
    ```bash
    pip install pipenv
    ```

3. Install the required packages using `pipenv`:
    ```bash
    pipenv install
    ```

4. Activate the virtual environment:
    ```bash
    pipenv shell
    ```

5. Download the spaCy model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

1. Place your dataset (`unredactor.tsv`) in the `resources` directory.

2. Run the `unredactor.py` script:
    ```bash
    pipenv run python unredactor.py
    ```

## Video Demonstration: [video](https://github.com/user-attachments/assets/08f0e00d-0d33-42d3-8d22-b6ce24e26a32)

## Functions

### `extract_features_with_sentiment(df, tfidf_vectorizer, dict_vectorizer=None)`

Extracts various features from the redacted text, including TF-IDF embeddings, sentiment scores, contextual features, positional features, and statistical features.

- **Parameters**:
  - `df`: DataFrame containing the redacted text.
  - `tfidf_vectorizer`: Fitted TF-IDF vectorizer.
  - `dict_vectorizer`: (Optional) Fitted DictVectorizer.

- **Returns**:
  - `feature_dicts`: List of feature dictionaries.
  - `dict_vectorizer`: Fitted DictVectorizer.
  - `vectorized_features`: Array of vectorized features.

### `train_model(vectorized_features, training_df)`

Trains a RandomForestClassifier model using the provided vectorized features and training data.

- **Parameters**:
  - `vectorized_features`: Array of vectorized features.
  - `training_df`: DataFrame containing the training data.

- **Returns**:
  - `model`: Trained RandomForestClassifier model.

### `evaluate_model(rf_model, validation_df, vectorized_features_val)`

Evaluates the trained model on the validation data and prints the evaluation metrics.

- **Parameters**:
  - `rf_model`: Trained RandomForestClassifier model.
  - `validation_df`: DataFrame containing the validation data.
  - `vectorized_features_val`: Array of vectorized features for the validation data.

### `dump_predicitons_to_file(rf_model, vectorized_features_val)`

Generates predictions using the trained model and writes them to a file.

- **Parameters**:
  - `rf_model`: Trained RandomForestClassifier model.
  - `vectorized_features_val`: Array of vectorized features for the validation data.


## Pytest Functions

- **`test_extract_features_with_sentiment`**: Tests the `extract_features_with_sentiment` function to ensure it returns the correct number of feature dictionaries and the correct shape of the vectorized features.
- **`test_train_model`**: Tests the `train_model` function to ensure it trains the model correctly and saves it to a file.
- **`test_evaluate_model`**: Tests the `evaluate_model` function to ensure it calculates the evaluation metrics correctly and that the metrics are greater than zero.

2. Run the tests:
    ```bash
    pipenv run python -m pytest
    ```

## Bugs, Issues and Assumptions

### Known Bugs

1. **Index Out of Bounds**: The code may raise an `IndexError` if the index of the TF-IDF embeddings array is accessed out of bounds. This can occur if the DataFrame has more rows than the TF-IDF embeddings array. Ensure that the TF-IDF embeddings are generated for the entire DataFrame.

2. **API Rate Limits**: The Google Knowledge Graph API has rate limits. If the dataset contains many unique names, the API calls may be rate-limited, causing delays. To mitigate this, there is no call for Knowledge Graph Search API.

3. **Performance Issues**: The feature extraction process can be time-consuming, especially for large datasets. The code uses parallel API calls and caching to improve performance, but further optimization may be needed for very large datasets.

4. **Data Quality Issues**: If the input data contains missing or malformed entries, the feature extraction process may fail or produce incorrect results. Ensure that the input data is clean and well-formatted.

5. **Model Overfitting**: The RandomForestClassifier may overfit the training data, especially if the dataset is small. This can lead to poor generalization on unseen data. Consider using cross-validation and hyperparameter tuning to mitigate overfitting.

6. **Dependency Conflicts**: Conflicts between different versions of dependencies (e.g., spaCy, scikit-learn, NLTK) can cause unexpected errors. Ensure that the correct versions of dependencies are installed as specified in the `Pipfile` or `requirements.txt`.

7. **Resource Limitations**: Processing large datasets can consume significant memory and CPU resources, potentially leading to resource exhaustion. Consider optimizing the code for memory and CPU usage or processing the data in smaller batches.

8. **Inconsistent Feature Extraction**: If the feature extraction process changes between training and testing, the model may fail to make accurate predictions. Ensure that the same feature extraction logic is applied consistently.

9. **Serialization Issues**: When saving and loading models using `joblib`, ensure that the environment (e.g., Python version, library versions) remains consistent. Inconsistencies can lead to deserialization errors.

10. **Error Handling**: The code may not handle all possible exceptions gracefully, leading to crashes or undefined behavior. Implement robust error handling to manage unexpected situations.


### Assumptions

- The dataset (`unredactor.tsv`) is a tab-separated file with three columns: `training_validation`, `names`, and `redacted_text`.
- The `training_validation` column indicates whether the row is for training or validation.
- The `names` column contains the original names that have been redacted in the `redacted_text` column.
- The redacted text uses the character `█` to indicate redacted portions.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License.

## Contact

For any questions or issues, please open an issue on GitHub or contact the project maintainer at [prathamrao18092001@gmail.com].
