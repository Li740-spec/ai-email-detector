import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier # 改用森林模型，判斷力更強
import os

def extract_features(text):
    # 這是在幫 AI 建立「判斷邏輯」
    features = {
        'has_link': 1 if 'http' in text or '連結' in text or '點此' in text else 0,
        'has_urgent': 1 if any(word in text for word in ['立即', '緊急', '過期', '分鐘內', '否則']) else 0,
        'has_bank': 1 if any(word in text for word in ['銀行', '帳戶', '登入', '驗證']) else 0
    }
    return features

def train_mixed_models():
    # 增加更多樣化的對照組
    raw_data = [
        ("【官方】您的帳戶已成功登入，若非本人請留意", "Safe Email"),
        ("緊急！您的帳戶被凍結，請立即點擊連結重置：http://fake.it", "Phishing Email"),
        ("明天下午的會議記錄請查收", "Safe Email"),
        ("Netflix 付款失敗，請點此更新資訊", "Phishing Email"),
        ("您有未領取的包裹，請支付 50 元運費", "Phishing Email"),
        ("感謝您的購物，訂單編號 12345 已出貨", "Safe Email")
    ]
    
    # 資料放大
    df = pd.DataFrame(raw_data * 100, columns=['Email Text', 'Email Type'])
    df['label'] = df['Email Type'].map({'Phishing Email': 1, 'Safe Email': 0})

    print(f"🧠 正在教 AI 讀懂語氣模式... (樣本數: {len(df)})")

    # 1. 依然使用 TF-IDF 處理文字
    vec_zh = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    X_text = vec_zh.fit_transform(df['Email Text'])
    
    # 2. 改用 RandomForest (隨機森林)
    # 這種模型會像「決策樹」一樣問問題：有連結嗎？語氣急嗎？是銀行嗎？
    clf_zh = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    clf_zh.fit(X_text, df['label'])

    # 存檔
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(clf_zh, 'models/clf_zh.joblib')
    joblib.dump(vec_zh, 'models/tfidf_zh.joblib')
    joblib.dump(clf_zh, 'models/clf_en.joblib')
    joblib.dump(vec_zh, 'models/tfidf_en.joblib')
    
    print(f"✨ 進階智慧模型訓練完成！")

if __name__ == "__main__":
    train_mixed_models()