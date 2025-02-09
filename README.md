# Semantic Book Recommender

The **Semantic Book Recommender** is a machine learning-powered project designed to recommend books based on their semantic relevance. It leverages Natural Language Processing (NLP) techniques to classify, analyze sentiments, and perform vector-based search for books, ensuring tailored recommendations for users.

## Folder Structure

```
.semantic-book-recommender
│
├── .gradio                 # Gradio configuration files
├── .idea                   # IDE-specific configuration files (e.g., PyCharm)
├── data                    # Dataset files and preprocessed data
│   ├── books_cleaned.csv
│   ├── books_cleaned_with_categories.csv
│   ├── books_cleaned_with_emotions.csv
│   ├── cover-not-found.jpg
│   └── tagged_description.txt
│
├── notebooks               # Jupyter Notebooks for exploratory and model development
│   ├── eda.ipynb           # Exploratory Data Analysis
│   ├── sentiment-analysis.ipynb  # Sentiment analysis implementation
│   ├── text-classification.ipynb # Text classification implementation
│   └── vector-search.ipynb       # Vector search-based recommendation system
│
├── sbr-env                 # Virtual environment files
│
├── scripts                 # Python scripts for key functionality
│   ├── __init__.py         # Initialization for the scripts module
│   └── gradio-dashboard.py # Gradio-based user interface
│
├── src                     # Core source code for the project
│   └── __init__.py         # Initialization for the source code module
│
├── tests                   # Unit tests for the project
├── .env                    # Environment variables (excluded from version control)
├── .gitignore              # Git ignore file
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
```

## Features

- **Data Preprocessing:**
  - Processed datasets for semantic analysis, including category and emotion annotations.

- **Exploratory Data Analysis (EDA):**
  - Comprehensive analysis of book data to uncover patterns and insights.

- **Sentiment Analysis:**
  - Classify book descriptions and reviews based on sentiment.

- **Text Classification:**
  - Categorize books into predefined genres or topics using NLP models.

- **Vector Search for Recommendations:**
  - Implemented a semantic search engine for personalized book recommendations.

- **Interactive Dashboard:**
  - Built with Gradio for a user-friendly interface to interact with the recommender system.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yordanoswuletaw/semantic-book-recommender.git
   cd semantic-book-recommender
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Gradio dashboard:
   ```bash
   python scripts/gradio-dashboard.py
   ```

## Usage

- Access the Gradio dashboard through the provided URL.
- Upload book descriptions or search for books using keywords to receive recommendations.

## Datasets

The `data` folder contains cleaned datasets used for training and testing, including:
- `books_cleaned.csv`: Basic book data.
- `books_cleaned_with_categories.csv`: Books with category annotations.
- `books_cleaned_with_emotions.csv`: Books with emotional tags.
- `tagged_description.txt`: Sample tagged descriptions for vector search.

## Notebooks

The `notebooks` folder includes detailed workflows for:
- Data exploration (`eda.ipynb`)
- Sentiment analysis (`sentiment-analysis.ipynb`)
- Text classification (`text-classification.ipynb`)
- Semantic search (`vector-search.ipynb`)

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.