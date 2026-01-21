# MLOps Project: YouTube Comments Topic Modeling

This project implements an automated Machine Learning Operations (MLOps) pipeline designed to collect YouTube comments on a specific topic, analyze them, identify underlying themes (topic modeling), and predict these themes for new data. The entire process is orchestrated using **Apache Airflow** and utilizes **MLflow** for model tracking and versioning.

## Project Description

The primary goal of this project is to extract unstructured text data (YouTube comments), transform it into a structured format, and apply machine learning methods to automatically classify the content by topic.

Key features include:

*   **Automation**: Airflow DAGs run the training and prediction processes on a predefined schedule.
*   **Data Collection**: Integration with the YouTube API to fetch fresh comments based on a keyword (`psychology`).
*   **Text Processing**: Robust cleaning and lemmatization tailored for the Russian language.
*   **Modeling**: Application of **spectral clustering** to generate topic labels (as the data is unlabelled) and subsequent training of a **logistic regression** classifier to predict these topics.
*   **MLOps**: Utilizing **MLflow** for logging parameters, metrics, and versioning key models (TF-IDF vectorizer and classifier).
*   **Configuration**: All parameters are centrally managed via a single `params_all.yaml` file.

## Project Structure

The standard hierarchical structure ensures modularity and clean code separation.

```text
├── config/
│ └── params_all.yaml         # Configuration file for train/predict stages
├── dags/
│ └── test_train_dag.py       # Apache Airflow DAG definitions
├── data/
│ └── text.csv                # Output file with top words per topic
├── notebooks/
│ └── Topic modeling.ipynb    # Jupyter notebook for EDA/prototyping
├── src/
│ ├── cluster_train.py        # Clustering logic and metric calculations
│ ├── get_comments.py         # YouTube API data collection module
│ └── preprocessing_text.py   # Text cleaning and lemmatization functions
├── .gitignore
├── predict.py                # Inference (prediction) script
└── train.py                  # Model training script
```


## Technologies Used

| Category | Tool/Library | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | `Apache Airflow` | Scheduling and running tasks (`train`, `predict`). |
| **MLOps** | `MLflow` | Experiment tracking, model registration, and versioning. |
| **Data Handling** | `Python`, `pandas`, `numpy` | Core data manipulation logic. |
| **NLP** | `nltk`, `pymystem3` (`Mystem`) | Stopword removal, Russian language lemmatization. |
| **Modeling** | `scikit-learn` | TF-IDF, Spectral Clustering, Logistic Regression. |
| **API** | `requests`, `pyyoutube` | Interaction with the YouTube Data API v3. |
| **Configuration** | `PyYAML` | Managing project parameters. |

## Pipeline Overview (Workflow)

1.  **Orchestration (`dags/test_train_dag.py`)**: Airflow triggers `train.py` at 01:00 AM hourly and `predict.py` at 03:00 AM hourly.
2.  **Data Collection (`src/get_comments.py`)**: Both scripts fetch fresh comments from YouTube using the query "психология".
3.  **Preprocessing (`src/preprocessing_text.py`)**: Text is cleaned of emojis/links and lemmatized using Mystem.
4.  **Training (`train.py`, `src/cluster_train.py`)**:
    *   Data is vectorized (TF-IDF).
    *   Optimal topic count is automatically determined via spectral clustering and silhouette score evaluation.
    *   A classifier (`LogisticRegression`) is trained to predict these discovered topics.
    *   Models and metrics are logged to MLflow.
    *   Model version numbers are updated in `params_all.yaml`.
5.  **Prediction (`predict.py`)**:
    *   The latest model versions are loaded from MLflow.
    *   New comments are classified.
    *   The top 10 keywords for each identified topic are saved to `data/text.csv`.
