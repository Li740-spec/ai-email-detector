import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import os

def train_mixed_models():
    # 1. 基礎種子資料 (你可以自行增加)
    data = [
        ("您的帳戶密碼已過期，請點擊連結重置", "Phishing Email"),
        ("恭喜獲得星巴克禮卷，請輸入個資領取", "Phishing Email"),
        ("Netflix 會員扣款失敗，請更新信用卡資訊", "Phishing Email"),
        ("國立高雄科技大學 校友中心 通知", "Safe Email"),
        ("這是您的電子發票開立通知，請查收", "Safe Email"),
        ("感謝您參與本校就業博覽會活動", "Safe Email")
    ]
    
    # 2. 讀取回饋資料 (feedback.csv)
    if os.path.exists('feedback.csv'):
        print("📈 發現糾正資料庫，正在整合學習...")
        fb_df = pd.read_csv('feedback.csv')
        fb_list = list(zip(fb_df['Email Text'], fb_df['Email Type']))
        # 權重設為 3，既能學會又不至於過度敏感
        data.extend(fb_list * 3) 

    # 3. 轉換資料格式
    df = pd.DataFrame(data, columns=['Email Text', 'Email Type'])
    df['label'] = df['label'] = df['Email Type'].apply(lambda x: 1 if x == 'Phishing Email' else 0)

    # 4. 特徵提取與訓練
    vec_zh = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    X_text = vec_zh.fit_transform(df['Email Text'])
    
    clf_zh = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf_zh.fit(X_text, df['label'])

    # 5. 存檔 (包含中英文名以相容舊腳本)
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(clf_zh, 'models/clf_zh.joblib')
    joblib.dump(vec_zh, 'models/tfidf_zh.joblib')
    joblib.dump(clf_zh, 'models/clf_en.joblib')
    joblib.dump(vec_zh, 'models/tfidf_en.joblib')
    
    print(f"⚖️ AI 進化完成！總學習樣本數: {len(df)}")

if __name__ == "__main__":
    train_mixed_models()