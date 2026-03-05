import os, re, joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

PROJECT = os.path.expanduser('~/Desktop/phishing_agent')
DATA_DIR = os.path.join(PROJECT, 'data')
MODEL_DIR = os.path.join(PROJECT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

csvs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.csv')]
if not csvs:
    raise SystemExit("No CSV found in data/")

df = pd.read_csv(os.path.join(DATA_DIR, csvs[0]))
if 'text' not in df.columns:
    if 'subject' in df.columns and 'body' in df.columns:
        df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    elif 'body' in df.columns:
        df['text'] = df['body'].fillna('')
    else:
        raise SystemExit("CSV missing text column")

if 'label' not in df.columns:
    raise SystemExit("CSV must have label column")

def clean(s):
    if pd.isna(s): return ''
    s = re.sub(r'<[^>]+>', ' ', str(s))
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

df['text'] = df['text'].apply(clean)
df = df[df['text'].str.strip()!='']

X = df['text'].values
y = df['label'].astype(int).values

# split first (do not leak)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=(y if len(set(y))>1 else None))

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

# SMOTE needs dense array
sm = SMOTE(random_state=42)
Xtr_res, ytr_res = sm.fit_resample(Xtr, y_train)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr_res, ytr_res)
y_pred = clf.predict(Xte)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(clf, os.path.join(MODEL_DIR, 'baseline_clf_smote.joblib'))
joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer_smote.joblib'))
print("Saved SMOTE model to", MODEL_DIR)
