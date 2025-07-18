import os
import pandas as pd
import re
import joblib
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Buat folder model jika belum ada
os.makedirs('model', exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv('data/data_dengan_sentimen.csv')

# Cek missing values
if df.isnull().values.any():
    print("Warning: Missing values found. Dropping...")
    df.dropna(inplace=True)

# Preprocessing
stop_words = set(stopwords.words('indonesian') + stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text))
    text = re.sub(r'[^a-z\s]', '', text.lower())
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

print("Preprocessing data...")
df['clean_text'] = df['tweet'].apply(preprocess_text)

# Features & Labels
X = df['clean_text']
y = df['Sentiment']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data (X masih teks!)
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# TF-IDF Vectorizer (akan dipakai di pipeline)
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

def create_pipeline(model):
    """
    Create an imblearn pipeline:
    - TF-IDF
    - SMOTE
    - Classifier
    """
    return ImbPipeline([
        ('tfidf', tfidf_vectorizer),
        ('smote', SMOTE(random_state=42)),
        ('clf', model)
    ])

# ========== NAIVE BAYES ==========
print("\nTraining Naive Bayes...")
nb_pipeline = create_pipeline(MultinomialNB())
nb_pipeline.fit(X_train, y_train_encoded)

y_pred_nb = nb_pipeline.predict(X_test)

print("\nNaive Bayes Classification Report:")
print(classification_report(
    y_test_encoded,
    y_pred_nb,
    target_names=le.classes_
))

cm_nb = confusion_matrix(y_test_encoded, y_pred_nb)
plt.figure(figsize=(8,6))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Naive Bayes Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ========== SVM ==========
print("\nTraining SVM...")
svm_model = SVC(random_state=42, probability=True)
svm_pipeline = create_pipeline(svm_model)
svm_pipeline.fit(X_train, y_train_encoded)

y_pred_svm = svm_pipeline.predict(X_test)

print("\nSVM Classification Report:")
print(classification_report(
    y_test_encoded,
    y_pred_svm,
    target_names=le.classes_
))

cm_svm = confusion_matrix(y_test_encoded, y_pred_svm)
plt.figure(figsize=(8,6))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ========== XGBOOST ==========
print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False
)
xgb_pipeline = create_pipeline(xgb_model)
xgb_pipeline.fit(X_train, y_train_encoded)

y_pred_xgb = xgb_pipeline.predict(X_test)

print("\nXGBoost Classification Report:")
print(classification_report(
    y_test_encoded,
    y_pred_xgb,
    target_names=le.classes_
))

cm_xgb = confusion_matrix(y_test_encoded, y_pred_xgb)
plt.figure(figsize=(8,6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ========== SAVE MODEL ==========
joblib.dump(nb_pipeline, 'model/nb_pipeline.pkl')
joblib.dump(svm_pipeline, 'model/svm_pipeline.pkl')
joblib.dump(xgb_pipeline, 'model/xgb_pipeline.pkl')
joblib.dump(tfidf_vectorizer, 'model/tfidf_vectorizer.pkl')
joblib.dump(le, 'model/label_encoder.pkl')

# Simpan test data agar Streamlit bisa pakai
joblib.dump(
    (X_test, y_test_encoded),
    'model/test_data.pkl'
)

print("\nTraining finished. Models saved successfully.")
