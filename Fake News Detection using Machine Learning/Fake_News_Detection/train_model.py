import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from preprocessing import preprocess_text

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FAKE_CSV = os.path.join(DATASET_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATASET_DIR, "True.csv")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
COMPARISON_PATH = os.path.join(BASE_DIR, "models_comparison.pkl")


def load_and_prepare_dataset():
    """
    Loads Fake.csv and True.csv, assigns binary labels, merges title + text,
    shuffles data, and pre-processes text.
    """
    print("[1/5] Loading datasets...")
    if not os.path.exists(FAKE_CSV) or not os.path.exists(TRUE_CSV):
        raise FileNotFoundError(f"Missing dataset files in {DATASET_DIR}. Ensure Fake.csv and True.csv exist.")

    df_fake = pd.read_csv(FAKE_CSV)
    df_true = pd.read_csv(TRUE_CSV)

    # Assign labels: Fake = 0, Real = 1
    df_fake['label'] = 0
    df_true['label'] = 1

    # Combine datasets
    df = pd.concat([df_fake, df_true], ignore_index=True)

    # Fill NaN values
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')

    # Merge title and text
    df['combined_text'] = df['title'] + " " + df['text']

    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Total articles loaded: {len(df)} (Fake: {sum(df['label']==0)}, Real: {sum(df['label']==1)})")
    
    print("[2/5] Cleaning and preprocessing text (lower, punctuation removal, stopwords, Porter stemming)...")
    df['processed_text'] = df['combined_text'].apply(preprocess_text)

    return df


def train_and_evaluate():
    """
    Main training workflow:
    - Preprocess data
    - Extract TF-IDF features
    - Split data (80:20)
    - Train Logistic Regression, Naive Bayes, and Random Forest
    - Evaluate performance and save joblib binaries & metrics.json
    """
    df = load_and_prepare_dataset()

    X = df['processed_text']
    y = df['label']

    print("[3/5] Extracting features using TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X)

    print("[4/5] Splitting dataset into 80% Train and 20% Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    trained_models = {}

    print("[5/5] Training and evaluating models...")
    for name, model in models.items():
        print(f"  --> Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate probabilities for ROC AUC if supported
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = 0.0
            fpr, tpr = [], []

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary')
        rec = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        cm = confusion_matrix(y_test, y_pred).tolist()
        cls_report = classification_report(y_test, y_pred, target_names=["FAKE (0)", "REAL (1)"], output_dict=True)

        results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "confusion_matrix": cm,
            "classification_report": cls_report,
            "roc_auc": float(roc_auc),
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr]
        }
        trained_models[name] = model

        print(f"      {name} -> Accuracy: {acc*100:.2f}%, F1-Score: {f1*100:.2f}%")

    # Save primary model (Logistic Regression) and vectorizer
    joblib.dump(trained_models["Logistic Regression"], MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    # Save all comparison models
    joblib.dump(trained_models, COMPARISON_PATH)

    # Save metrics summary to JSON
    summary = {
        "dataset_summary": {
            "total_articles": len(df),
            "fake_count": int(sum(df['label'] == 0)),
            "real_count": int(sum(df['label'] == 1))
        },
        "model_metrics": results
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "="*50)
    print("SUCCESS: Model training and evaluation complete!")
    print(f"  - Primary Model saved to: {MODEL_PATH}")
    print(f"  - Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"  - Evaluation metrics saved to: {METRICS_PATH}")
    print("="*50)

if __name__ == "__main__":
    train_and_evaluate()
