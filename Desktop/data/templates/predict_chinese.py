import joblib, os, jieba

MODEL_DIR = "models"
THRESHOLD = 0.65

# 載入模型
clf = joblib.load(os.path.join(MODEL_DIR, 'chinese_clf_smote.joblib'))
tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer_chinese.joblib'))

def chinese_tokenizer(text):
    return list(jieba.cut(text))

samples = [
    "您好，附件是本週會議議程，請查收",
    "請點擊以下連結確認您的帳號資訊 http://phish.example",
    "恭喜加入我們團隊，入職資料請查收",
]

for s in samples:
    # 中文斷詞後向量化
    v = tfidf.transform([s])
    prob = clf.predict_proba(v)[0][1]
    pred = 1 if prob >= THRESHOLD else 0
    print(f"TEXT: {s}")
    print(f" -> prob(class=1) = {prob:.3f}  -> pred = {pred}\n")
