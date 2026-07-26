# 📩 SMS / Email Spam Classifier

A machine learning web app that detects whether a text message (SMS or Email) is **Spam** or **Not Spam**, built with **Scikit-Learn**, **NLTK**, and **Streamlit**.

<p align="center">
  <img alt="App Banner" src="https://github.com/user-attachments/assets/74f36f53-0a08-4b3d-973a-40d0dd9c02d2" width="700"/>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Live Demo"></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/NLTK-Text%20Processing-green" alt="NLTK">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</p>

## 🔗 Live Demo

**Try it here 👉 [Live App on Streamlit Community Cloud](YOUR_STREAMLIT_APP_LINK_HERE)**

> Replace the link above with your app's URL once it's deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).

---

## 📸 Screenshots

🚨 Spam Detected

<img width="667" height="750" alt="Not spam screenshot" src="https://github.com/user-attachments/assets/444cda60-cb9d-4771-8cf1-b024be61186e" />


✅ Not Spam

<img width="672" height="800" alt="Spam detected screenshot" src="https://github.com/user-attachments/assets/5e66994e-f772-4dbd-ba8d-13beeae8f0ce" />

---

## 📝 Overview

This project classifies SMS/Email messages as **Spam** or **Ham (Not Spam)** using a text-preprocessing pipeline combined with a **Multinomial Naive Bayes** classifier trained on a TF-IDF representation of the text. The trained model is served through an interactive **Streamlit** web app where a user can paste any message and get an instant prediction along with a confidence score.

## ✨ Features

- 🔍 Real-time spam/ham prediction with confidence score
- 🧹 Full text-preprocessing pipeline (lowercasing, tokenization, stopword removal, punctuation removal, stemming)
- 📊 Exploratory Data Analysis notebook included (class distribution, message length stats, word clouds)
- 🤖 Benchmarked 11 ML algorithms before selecting the final model
- 🖥️ Clean, responsive Streamlit UI
- ☁️ Ready for one-click deployment on Streamlit Community Cloud

## 🗂️ Project Structure

```
SMS-Spam-Classifier/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore
│
├── models/
│   ├── model.pkl                   # Trained Multinomial Naive Bayes model
│   └── vectorizer.pkl              # Fitted TF-IDF vectorizer
│
├── data/
│   └── spam.csv                    # Raw SMS Spam Collection dataset
│
├── notebooks/
│   └── SMS-spam_classifier.ipynb   # EDA, preprocessing & model training notebook
│
└── assets/
    └── app_screenshot.png          # App screenshots
```

## 📊 Dataset

The model is trained on the **SMS Spam Collection Dataset** (`spam.csv`), which contains **5,572 labeled messages**:

- After removing duplicates, **5,169** unique messages remained
- **4,516** Ham messages and **653** Spam messages (imbalanced dataset)
- Columns used: `target` (ham/spam) and `text` (the message content)

## 🧠 Approach

1. **Data Cleaning** – dropped unused columns, removed duplicate rows, label-encoded the target
2. **EDA** – analyzed character/word/sentence counts, visualized distributions and correlations, generated word clouds for spam vs. ham
3. **Text Preprocessing** (`transform_text` function):
   - Lowercasing
   - Tokenization (`nltk.word_tokenize`)
   - Removing non-alphanumeric tokens
   - Removing English stopwords and punctuation
   - Stemming with `PorterStemmer`
4. **Feature Extraction** – `TfidfVectorizer` (top 3,000 features)
5. **Model Selection** – trained and compared **11 algorithms** (Naive Bayes variants, SVM, Logistic Regression, Random Forest, KNN, Decision Tree, AdaBoost, Bagging, Extra Trees, Gradient Boosting, XGBoost), and also tested Voting/Stacking ensembles
6. **Final Model** – **Multinomial Naive Bayes** was selected since precision (avoiding false spam flags on genuine messages) was the priority metric and it achieved a perfect precision score with strong accuracy

## 📈 Model Performance

| Metric | Score |
|---|---|
| Accuracy | **97.1%** |
| Precision | **100%** |

> Precision was prioritized over accuracy for this problem — a false positive (marking a real message as spam) is far more costly than an occasional missed spam message.

## 🛠️ Tech Stack

- **Language:** Python 3
- **Machine Learning:** Scikit-Learn, NLTK
- **Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn, WordCloud
- **Web App:** Streamlit
- **Model Persistence:** Pickle

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/SMS-Spam-Classifier.git
cd SMS-Spam-Classifier

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

## 🔮 Future Improvements

- Handle class imbalance with techniques like SMOTE
- Experiment with word embeddings (Word2Vec, GloVe) or transformer-based models (BERT)
- Add a REST API endpoint for programmatic access
- Add multilingual spam detection support

## 📄 License

This project is licensed under the **MIT License**.

## 🙌 Author

**Shubham Malkar**

Built with ❤️ using Streamlit, Scikit-Learn & NLTK.
