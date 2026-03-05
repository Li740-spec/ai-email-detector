import joblib, os
os.chdir(os.path.expanduser('~/Desktop/phishing_agent'))
clf = joblib.load('models/baseline_clf.joblib')
tfidf = joblib.load('models/tfidf_vectorizer.joblib')

threshold = 0.4   # 把這個數字調低會標更多釣魚（提高 recall）
samples = [
    "Please confirm your payment method: http://secure-pay.example",
    "Let's have lunch tomorrow.",
    "Your account has been suspended. Click here to verify your identity.",
    "Invoice attached — please process payment."
]

for s in samples:
    v = tfidf.transform([s.lower()])
    prob = clf.predict_proba(v).max() if hasattr(clf,'predict_proba') else None
    pred = 1 if (prob is not None and prob >= threshold) else 0
    print(s)
    print("prob(max)=", prob, " -> pred(threshold=%s)=" % threshold, pred)
    print()
