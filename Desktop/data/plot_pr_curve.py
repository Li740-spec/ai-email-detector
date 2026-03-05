import joblib, os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
os.chdir(os.path.expanduser('~/Desktop/phishing_agent'))

# load tfidf & model (use SMOTE model if exists else baseline)
if os.path.exists('models/baseline_clf_smote.joblib'):
    clf = joblib.load('models/baseline_clf_smote.joblib')
    tfidf = joblib.load('models/tfidf_vectorizer_smote.joblib')
else:
    clf = joblib.load('models/baseline_clf.joblib')
    tfidf = joblib.load('models/tfidf_vectorizer.joblib')

# load data
import pandas as pd
df = pd.read_csv('data/sample_phish.csv')
X = df['text'].fillna('').astype(str).str.lower().values
y = df['label'].astype(int).values

# transform
Xv = tfidf.transform(X)
probs = clf.predict_proba(Xv)[:,1]
precision, recall, thresholds = precision_recall_curve(y, probs)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'PR curve (AUC={pr_auc:.3f})')
plt.grid(True)
plt.savefig('pr_curve.png')
print('Saved pr_curve.png')
