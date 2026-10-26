# Patient Case Semantic Search System

An AI-powered information retrieval system that uses **semantic embeddings** to search and retrieve clinically relevant patient cases based on natural-language queries.

Unlike traditional keyword-based search, this project uses Sentence Transformers to represent clinical notes as numerical vectors and identify cases with similar meanings.

## Project Overview

Finding relevant clinical notes using simple keyword matching can be challenging because different medical descriptions may express similar symptoms using different words.

This project addresses that challenge using semantic similarity search. Users can enter a symptom-related query and retrieve the most similar clinical notes from a dataset of 2,000 synthetic clinical records.

Important: This project is intended for educational and information-retrieval purposes only. It is not a diagnostic tool and must not be used to make medical decisions.

## Key Features

* Semantic search over 2,000 clinical notes.
* Natural-language search queries.
* Sentence embeddings using `all-MiniLM-L6-v2`.
* Cosine similarity for ranking retrieved cases.
* Interactive web interface built with Streamlit.
* Data cleaning and exploratory analysis.
* Retrieval evaluation using Precision@5 and Mean Reciprocal Rank (MRR).

## Technologies Used

* Python
* Pandas and NumPy
* Sentence Transformers
* Scikit-learn
* Streamlit
* Matplotlib
* Jupyter Notebook
* Hugging Face Datasets

## Dataset

The project uses a sample of 2,000 records from the Hugging Face dataset:

Asclepius Synthetic Clinical Notes
Dataset: `aisc-team-a1/Asclepius-Synthetic-Clinical-Notes`

The sample contains fields such as patient ID, clinical note, question, answer, and task. The search system uses the cleaned clinical notes for retrieval.

The dataset is synthetic. Please consult the original dataset page for its license and attribution requirements before redistributing the data or using it for other purposes.

## How It Works

1. **Data collection:** Obtain a sample of synthetic clinical notes.
2. **Data preprocessing:** Remove missing or blank notes and duplicate note text.
3. **Embedding generation:** Convert each clinical note into a vector using Sentence Transformers.
4. **Query processing:** Convert the user's natural-language query into an embedding using the same model.
5. **Similarity calculation:** Calculate cosine similarity between the query embedding and note embeddings.
6. **Ranking:** Sort clinical notes by similarity score.
7. **Retrieval:** Display the top matching notes in the Streamlit interface.
8. **Evaluation:** Assess the top-five search results using manually assigned relevance labels.

## Model

The project uses:

`sentence-transformers/all-MiniLM-L6-v2`

This pretrained Sentence Transformer generates sentence-level embeddings that can be compared to retrieve semantically similar text.

The system uses cosine similarity to rank candidate notes against a search query.

## Evaluation Results

The system was evaluated using five symptom-related search queries, with five results retrieved for each query.

| Evaluation Metric            | Result |
| ---------------------------- | -----: |
| Mean Precision@5             |   0.80 |
| Mean Reciprocal Rank (MRR)   |   1.00 |
| Number of evaluation queries |      5 |
| Results per query            |      5 |

### Interpretation

* **Mean Precision@5 = 0.80:** On average, four of the five retrieved results per query were marked relevant.
* **MRR = 1.00:** The first retrieved result was marked relevant for each of the five evaluation queries.

These are preliminary results based on manual relevance judgments over a small set of queries. Performance may vary with different queries and datasets.

Detailed evaluation files are saved in the `data/` directory.

## Project Structure

```text
patient-case-semantic-search/
├── data/
│   ├── patient_cases.csv
│   ├── patient_cases_large.csv
│   ├── patient_cases_cleaned.csv
│   ├── evaluation_results.csv
│   └── evaluation_metrics.csv
├── notebooks/
│   └── patient_case_exploration.ipynb
├── src/
│   ├── __init__.py
│   └── semantic_search.py
├── app.py
├── evaluate_search.py
├── calculate_metrics.py
├── stream_dataset.py
├── check_dataset.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd patient-case-semantic-search
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the URL of your GitHub repository.

### 2. Create and activate a Python environment

Using Conda:

```bash
conda create -n patient_search python=3.11
conda activate patient_search
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the cleaned dataset at:

```text
data/patient_cases_cleaned.csv
```

The application expects this file to exist before starting the search interface.

### 5. Run the application

From the project root directory:

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser to use the application.

## Running the Evaluation

To generate the retrieval results:

```bash
python evaluate_search.py
```

After assigning relevance labels (`1` for relevant and `0` for not relevant) in `data/evaluation_results.csv`, calculate the evaluation metrics:

```bash
python calculate_metrics.py
```

The metrics will be saved to:

```text
data/evaluation_metrics.csv
```

## Future Improvements

* Expand the evaluation to include more queries.
* Compare semantic search against a keyword-based baseline.
* Add Recall@K and ranking metrics where appropriate.
* Experiment with alternative embedding models.
* Improve search speed through embedding caching or a vector index.
* Add filters for clinical topics and note attributes.
* Deploy the application online.
* Add stronger evaluation protocols and expert review before considering any clinical use.

## License and Responsible Use

This repository contains an educational information-retrieval project. The original dataset has its own license and attribution requirements; consult its official terms before redistributing dataset records.

Do not use this application to diagnose, treat, or make decisions about real patients.

## Author

Usmita Banerjee
MSc Data Science
