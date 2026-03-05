import os, sys
proj = os.path.expanduser('~/Desktop/phishing_agent')
os.chdir(proj)
import joblib

MODEL_P = os.path.join('models','baseline_clf.joblib')
TFIDF_P = os.path.join('models','tfidf_vectorizer.joblib')

if not os.path.exists(MODEL_P) or not os.path.exists(TFIDF_P):
    print("Model or vectorizer not found in models/.")
    print("Run: python train_baseline.py  (to train and save baseline model and tfidf)")
    sys.exit(1)

clf = joblib.load(MODEL_P)
tfidf = joblib.load(TFIDF_P)

def predict(text):
    txt = text.lower()
    v = tfidf.transform([txt])
    pred = clf.predict(v)[0]
    prob = None
    if hasattr(clf, 'predict_proba'):
        try:
            prob = clf.predict_proba(v).max()
        except:
            prob = None
    return int(pred), float(prob) if prob is not None else None

samples = [
    "Please confirm your payment method: http://secure-pay.example",
    "Let's have lunch tomorrow.",
    "Your account has been suspended. Click here to verify your identity.",
    "Invoice attached — please process payment."
]

for s in samples:
    p, prob = predict(s)
    print("TEXT:", s)
    print(" -> PRED:", p, ("(prob={:.3f})".format(prob) if prob is not None else ""))
    print()
