"""
Train script for Fake News detection.

Usage:
1. Place a CSV in backend/data/news.csv with columns: title, text, label
   - label should be 'REAL' or 'FAKE' (or 0/1 — script will normalize)
2. python train.py --data-path backend/data/news.csv
3. The saved model will be written to backend/saved_model/model.joblib
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from joblib import dump
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    # try to find columns
    cols = df.columns
    if 'text' not in cols and 'article' in cols:
        df.rename(columns={'article': 'text'}, inplace=True)
    if 'title' not in cols:
        df['title'] = ""
    if 'label' not in cols and 'truth' in cols:
        df.rename(columns={'truth': 'label'}, inplace=True)
    # combine title+text
    df['content'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).str.strip()
    df = df[df['content'].str.len() > 10].copy()
    # normalize label
    df['label'] = df['label'].astype(str).str.upper().map(lambda x: 'REAL' if x in ['REAL', '1', 'TRUE', 'T'] else 'FAKE')
    return df[['content', 'label']]

def build_pipeline():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')),
        ("clf", LogisticRegression(max_iter=1000, solver='liblinear'))
    ])
    return pipe

def main(args):
    df = load_data(args.data_path)
    X = df['content'].values
    y = df['label'].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)

    pipe = build_pipeline()
    print("Training model...")
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]  # if label encoding puts FAKE or REAL at index 1 depends; we will compute accuracy only
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy on test set: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, preds, target_names=le.classes_))

    # Save both pipeline and label encoder together as a little wrapper object
    model_obj = {
        "pipeline": pipe,
        "label_encoder": le
    }
    os.makedirs(os.path.join(os.path.dirname(__file__), "saved_model"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "saved_model", "model.joblib")
    dump(model_obj, out_path)
    print(f"Saved model to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV file with data")
    args = parser.parse_args()
    main(args)
