import os, re, joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

PROJECT = os.path.expanduser('~/Desktop/phishing_agent')
DATA_DIR = os.path.join(PROJECT, 'data')
MODEL_DIR = os.path.join(PROJECT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# find first csv in data/
csvs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.csv')]
if not csvs:
    raise SystemExit("No CSV found in data/. Put a CSV with columns 'text' and 'label' into data/")

df = pd.read_csv(os.path.join(DATA_DIR, csvs[0]))
# handle common column names
if 'text' not in df.columns:
    if 'subject' in df.columns and 'body' in df.columns:
        df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    elif 'body' in df.columns:
        df['text'] = df['body'].fillna('')
    else:
        raise SystemExit("CSV has no 'text' column and no 'subject'/'body' pair. Please adjust file.")

if 'label' not in df.columns:
    raise SystemExit("CSV must have a 'label' column (1=phish,0=normal).")

# basic clean
def clean(s):
    if pd.isna(s): return ''
    s = re.sub(r'<[^>]+>', ' ', str(s))   # remove html tags
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

df['text'] = df['text'].apply(clean)
df = df[df['text'].str.strip()!='']

X = df['text'].values
y = df['label'].astype(int).values

# Robust train_test_split: try stratify, fallback to no stratify if too few samples
strat = y if len(set(y)) > 1 else None
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat)
except ValueError as e:
    print("Warning: stratified split failed (probably too few samples per class).")
    print("Details:", e)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr, y_train)
y_pred = clf.predict(Xte)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# save model + vectorizer
joblib.dump(clf, os.path.join(MODEL_DIR, 'baseline_clf.joblib'))
joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
print("Saved model to", MODEL_DIR)
