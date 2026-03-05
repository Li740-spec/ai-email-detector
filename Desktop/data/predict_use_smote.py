import joblib, os
os.chdir(os.path.expanduser('~/Desktop/phishing_agent'))
# use SMOTE model if exists, else baseline
if os.path.exists('models/baseline_clf_smote.joblib'):
    clf = joblib.load('models/baseline_clf_smote.joblib')
    tfidf = joblib.load('models/tfidf_vectorizer_smote.joblib')
    print("Using SMOTE model")
else:
    clf = joblib.load('models/baseline_clf.joblib')
    tfidf = joblib.load('models/tfidf_vectorizer.joblib')
    print("Using baseline model")

samples = [
    "Please confirm your payment method: http://secure-pay.example",
    "Let's have lunch tomorrow.",
    "Your account has been suspended. Click here to verify your identity.",
    "Invoice attached — please process payment."
]

for s in samples:
    v = tfidf.transform([s.lower()])
    prob = clf.predict_proba(v).max() if hasattr(clf,'predict_proba') else None
    pred = int(clf.predict(v)[0])
    print(s)
    print(" -> pred:", pred, " prob(max):", prob)
    print()
