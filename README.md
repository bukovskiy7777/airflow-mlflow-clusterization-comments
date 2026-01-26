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

## Results Interpretation: Identified Topics

The script processed YouTube comments (using "психология" as the query) and identified 9 distinct topics. The following table presents the top 10 keywords characterizing each theme, providing insight into the community's primary areas of interest. The topics are organized in columns as they appear in the output CSV file.

| Topic ID | Top Keywords | Interpretation |
| :--- | :--- | :--- |
| **0** | человек, власть, самый, гнилой, делать, жизнь, любовь, понимать, любить, бояться | **Core Human Concepts & Challenges**. This topic groups general philosophical and human existence terms ("human", "life", "love") mixed with concepts of power, criticism ("rotten"), and fear. It seems to represent a high-level, foundational theme about the human experience. **Основные понятия человеческого существования: жизнь, любовь, власть и страхи.** |
| **1** | видео, психолог, психология, канал, лекция, автор, глаз, тема, понимать, давно | **Content Consumption & Expert Analysis**. This theme focuses heavily on how users consume the content: videos, lectures, podcasts, and the role of the psychologist/author. It represents the active engagement with the media itself. **Обсуждение формата контента (видео, лекции) и роли автора/психолога.** |
| **2** | заботиться, лекция, лицо, любить, любовь, любой, мало, мальчик, мама, идея | **Emotional & Relational Dynamics**. This topic touches upon personal emotions and relationships, particularly family ("mama", "boy", "care", "love"). It suggests discussions around emotional well-being and personal connections. **Личные отношения, эмоции и семейные темы, включая заботу и любовь.** |
| **3** | благодарить, ролик, прекрасный, деньги, увидеть, работа, душа, информация, видео, слово | **Value Exchange & Information Utility**. This theme mixes expressions of gratitude ("thank") with the perceived value of the information ("useful information"), discussions about work, and potentially the financial aspects ("money") of expert advice. **Благодарность за полезную информацию, обсуждение работы и ценности контента.** |
| **4** | год, бабушка, любить, ребенок, мама, жизнь, детство, понимать, становиться, суп | **Generational & Life Stages**. This topic clearly groups terms related to time, age, family, and life stages ("grandmother", "child", "mother", "childhood", "year"). It likely covers content related to family dynamics and growing up. **Темы, связанные с возрастом, семьей, детством и жизненными этапами.** |
| **5** | спасибо, большой, подкаст, огромный, видео, жизнь, гость, полезный, тема, совет | **Feedback, Format, & Utility**. A strong theme of positive feedback ("thank you", "great") combined with specific content formats ("podcast", "guest") and the practical utility of the advice given ("useful", "advice"). **Положительные отзывы, обсуждение подкастов/гостей и практической пользы советов.** |
| **6** | слушать, книга, прямой, читать, приятно, психология, интересно, брать, минута, начинать | **Learning Process & Engagement**. This topic focuses on the active steps users take to learn: listening, reading books, "starting" a process, and finding content interesting or pleasant. **Процесс обучения через прослушивание/чтение, начало новых практик.** |
| **7** | женщина, мужик, мужчина, ребенок, особенно, готовить, делать, деньги, далеко, манипуляция | **Gender Roles, Action, & Criticism**. This theme seems to be more action-oriented ("do", "prepare") but also includes specific gender references ("woman", "man") and potentially darker or critical concepts like "money" and "manipulation". **Обсуждение гендерных ролей, действий и критические замечания (деньги, манипуляции).** |
| **8** | бог, мир, твой, самый, правда, слово, любовь, решать, посмотреть, взять | **Existential & Conclusive Concepts**. This final topic is highly abstract and conclusive: discussing "God", the "world", "truth", "love", and "solving" issues. It suggests a focus on existential themes or summarizing outcomes. **Экзистенциальные темы: поиск истины, мира, любви и решения проблем.** |


## 🧠 Deep Dive: ML Logic & Mathematical Base
The data processing workflow is divided into three key stages: text-to-numerical transformation, discovery of latent structures (clustering), and predictive model training.

**1. Vectorization: TF-IDF Transformation**

To enable algorithms to process text, we convert it into a vector space using TF-IDF (Term Frequency-Inverse Document Frequency).
* **Term Frequency (TF)**: Evaluates the importance of a word within a single comment.
* **Inverse Document Frequency (IDF)**: Reduces the weight of words that appear frequently across all comments (e.g., common stop words), highlighting terms specific to particular topics.

**Mathematical Base**:The weight of term $t$ in document $d$ is calculated as:

$$w_{t,d} = \text{tf}_{t,d} \times \log\left(\frac{N}{\text{df}_t}\right)$$
  
where $N$ is the total number of comments, and $\text{df}_t$ is the number of documents containing term $t$. This results in a sparse matrix where each comment is represented by a vector of significant features.
  
**2. Topic Discovery: Spectral Clustering & Silhouette Score**

Unlike traditional K-Means, **Spectral Clustering** utilizes the eigenvalues of similarity matrices to perform dimensionality reduction before clustering.

* **Why Spectral Clustering?** It excels at handling data where cluster boundaries have complex shapes, which is typical for natural language semantics.

* **Automated Topic Detection**: The number of topics is not set manually. Instead, the script iterates through a range of values and evaluates each using the **Silhouette Score**.

**Mathematical Base (Silhouette Score)**: For each object, the coefficient is calculated as:

$$s = \frac{b - a}{\max(a, b)}$$

Where:
* $a$ is the mean distance to objects within the same cluster.
* $b$ is the mean distance to objects in the nearest neighboring cluster.The value $s$ ranges from $[-1, 1]$. The script selects the number of clusters that maximizes the mean silhouette score across the entire dataset.

**3. Classification: Logistic Regression**

Once clustering has assigned a topic ID (Label) to each comment, we train a **Logistic Regression** classifier to predict these labels.
* **Why this step?** Clustering is computationally expensive. A trained classification model (Inference) operates almost instantaneously, allowing for the classification of new comments without recalculating the similarity matrix for the entire dataset.
* **Logic**: The model constructs separating hyperplanes in the high-dimensional TF-IDF feature space.

**Mathematical Base**: For multi-class classification, the **Softmax** function is used to transform model outputs into probabilities for each topic:

$$P(y=i | x) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

where $z_i$ is the linear combination of features for class $i$. The model minimizes the Cross-Entropy loss function to ensure the highest possible prediction accuracy.

## 📈 Metrics and Quality Control

During the training process, the following indicators are logged in **MLflow**:
* **Precision (Macro)**: Measures how accurately the model identifies a topic (minimizing false positives).
* **F1-Score (Macro)**: The harmonic mean of Precision and Recall, accounting for potential imbalances in the number of comments across different topics.
