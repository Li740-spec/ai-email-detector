import pandas as pd
import os, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import jieba

DATA_PATH = "data/emails.csv"
MODEL_DIR = "models"

# 讀資料
df = pd.read_csv(DATA_PATH)

# 中文斷詞函數
def chinese_tokenizer(text):
    return list(jieba.cut(text))

# TF-IDF (使用中文斷詞)
tfidf = TfidfVectorizer(tokenizer=chinese_tokenizer, ngram_range=(1,2))
X = tfidf.fit_transform(df['text'])
y = df['label']

#切訓練/測試集


# 平衡資料
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# 訓練模型
clf = LogisticRegression(max_iter=1000, C=0.5)
clf.fit(X_res, y_res)

# 評估
y_pred = clf.predict(X_res)
print("Accuracy:", accuracy_score(y_res, y_pred))
print(classification_report(y_res, y_pred))

# 儲存模型
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(clf, os.path.join(MODEL_DIR, 'chinese_clf_smote.joblib'))
joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer_chinese.joblib'))
print("Saved SMOTE Chinese model to", MODEL_DIR)
