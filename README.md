# Unredactor Project

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

## Video Demonstration: [video]

## Assumptions

- The dataset (`unredactor.tsv`) is a tab-separated file with three columns: `training_validation`, `names`, and `redacted_text`.
- The `training_validation` column indicates whether the row is for training or validation.
- The `names` column contains the original names that have been redacted in the `redacted_text` column.
- The redacted text uses the character `█` to indicate redacted portions.

## Bugs and Issues

### Known Bugs

1. **Index Out of Bounds**: The code may raise an `IndexError` if the index of the TF-IDF embeddings array is accessed out of bounds. This can occur if the DataFrame has more rows than the TF-IDF embeddings array. Ensure that the TF-IDF embeddings are generated for the entire DataFrame.

2. **API Rate Limits**: The Google Knowledge Graph API has rate limits. If the dataset contains many unique names, the API calls may be rate-limited, causing delays. To mitigate this, there is no call for Knowledge Graph Search API.  

3. **Performance Issues**: The feature extraction process can be time-consuming, especially for large datasets. The code uses parallel API calls and caching to improve performance, but further optimization may be needed for very large datasets.

### Assumptions

- The spaCy model (`en_core_web_sm`) and NLTK VADER lexicon are correctly installed.
- The dataset is correctly formatted and placed in the `resources` directory.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please open an issue on GitHub or contact the project maintainer at [prathamrao18092001@gmail.com].