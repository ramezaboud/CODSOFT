# CODSOFT Machine Learning Internship Tasks

This repository contains my completed tasks for the **CODSOFT Machine Learning Internship**. Each task focuses on different machine learning classification problems using Python.

## 📋 Table of Contents

- [Task 1: Movie Genre Classification](#task-1-movie-genre-classification)
- [Task 2: Credit Card Fraud Detection](#task-2-credit-card-fraud-detection)
- [Task 3: Customer Churn Prediction](#task-3-customer-churn-prediction)
- [Task 4: SMS Spam Detection](#task-4-sms-spam-detection)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Author](#author)

---

## 📁 Task 1: Movie Genre Classification

**File:** `task_1.ipynb`

A machine learning model that classifies movies into different genres based on their title and description.

### Features:
- Text preprocessing (cleaning, tokenization, lemmatization)
- TF-IDF vectorization for text features
- Multiple classification algorithms (Logistic Regression, SVM)
- Handling imbalanced data with SMOTE

### Dataset:
- Training data: `train_data.txt`
- Test data for predictions

---

## 📁 Task 2: Credit Card Fraud Detection

**File:** `task_2.ipynb`

A classification model to detect fraudulent credit card transactions.

### Features:
- Exploratory Data Analysis (EDA)
- Feature engineering and preprocessing
- Multiple classifiers comparison:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- Handling imbalanced data with SMOTE
- Model evaluation with confusion matrix and classification report

### Dataset:
- `fraudTrain.csv` - Training data
- `fraudTest.csv` - Test data

---

## 📁 Task 3: Customer Churn Prediction

**File:** `task_3.ipynb`

A predictive model to identify customers who are likely to churn (leave the service).

### Features:
- Comprehensive EDA with visualizations
- Feature encoding (OneHotEncoder, LabelEncoder)
- Feature scaling (MinMaxScaler, StandardScaler)
- Logistic Regression model
- Model evaluation metrics:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC Curve
  - Confusion Matrix

### Dataset:
- `Churn_Modelling.csv`

---

## 📁 Task 4: SMS Spam Detection

**File:** `task_4.ipynb`

A text classification model to detect spam SMS messages.

### Features:
- Text data preprocessing
- TF-IDF vectorization
- Logistic Regression classifier
- Model performance evaluation

### Dataset:
- `spam.csv`

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries:**
  - `pandas` - Data manipulation
  - `numpy` - Numerical computing
  - `matplotlib` & `seaborn` - Data visualization
  - `scikit-learn` - Machine learning algorithms
  - `nltk` - Natural Language Processing
  - `imbalanced-learn` - Handling imbalanced datasets

---

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/ramezaboud/CODSOFT.git
cd CODSOFT
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk imbalanced-learn
```

3. Download NLTK data (if needed):
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

4. Open Jupyter Notebook and run the tasks:
```bash
jupyter notebook
```

---

## 👤 Author

**Ramez Aboud**

- GitHub: [@ramezaboud](https://github.com/ramezaboud)

---

## 📄 License

This project is part of the CODSOFT Machine Learning Internship program.

---

⭐ If you found this helpful, please give it a star!
