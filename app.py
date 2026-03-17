import os, joblib, re
from flask import Flask, request, jsonify
from flask_cors import CORS
from difflib import SequenceMatcher

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 載入模型
try:
    clf = joblib.load(os.path.join(BASE_DIR, 'clf_zh.joblib'))
    tfidf = joblib.load(os.path.join(BASE_DIR, 'tfidf_zh.joblib'))
    print("✅ 專業版 AI 引擎載入成功")
except Exception as e:
    clf, tfidf = None, None
    print(f"❌ 載入失敗: {e}")

# 專業偵測資料庫
TRUSTED_BRANDS = ["google", "facebook", "apple", "microsoft", "amazon", "line", "netflix"]
MALICIOUS_DOMAINS = ["verify-g00gle.com", "login-microsoft.top", "account-update.xyz"]

def get_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

@app.route('/')
def home():
    return "AI Cybersecurity Server v2.0 - Online"

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS': return '', 200
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text: return jsonify({'label': 'safe', 'phish_prob': 0})
        
        # 1. 域名檢查
        urls = re.findall(r'https?://([a-zA-Z0-9.-]+)', text)
        spoofing = False
        on_blacklist = False
        for url in urls:
            domain = url.split('.')[0].lower()
            if any(bad in url for bad in MALICIOUS_DOMAINS): on_blacklist = True
            for brand in TRUSTED_BRANDS:
                sim = get_similarity(brand, domain)
                if 0.75 < sim < 1.0: spoofing = True

        # 2. AI 預測
        vec = tfidf.transform([text])
        ai_prob = float(clf.predict_proba(vec)[0][1])

        # 3. 意圖分類
        cats, boost = [], 0
        if re.search(r'登入|驗證|密碼|身分', text):
            cats.append("Credential Phishing"); boost += 0.2
        if re.search(r'立即|緊急|失效|限制', text):
            cats.append("Social Engineering"); boost += 0.2
        if urls:
            cats.append("Malicious Link"); boost += 0.1

        # 4. 整合權重
        final_prob = min(ai_prob + boost + (0.4 if spoofing else 0), 1.0)
        if on_blacklist: final_prob = 1.0

        # 5. 關鍵字清理
        feature_names = tfidf.get_feature_names_out()
        weights = vec.toarray()[0]
        stop_words = ['您的', '我們', '您可以', 'html', 'le', 'gle', 'title', 'meta']
        top_indices = weights.argsort()[::-1]
        keywords = [str(feature_names[i]) for i in top_indices 
                    if len(str(feature_names[i])) > 1 and str(feature_names[i]) not in stop_words][:3]

        return jsonify({
            'label': 'phishing' if final_prob >= 0.5 else 'safe',
            'phish_prob': round(final_prob * 100, 1),
            'threat_category': cats if cats else ["General Pattern"],
            'keywords': keywords,
            'spoofing_alert': spoofing,
            'advice': "🚩 高風險：偵測到域名偽裝！" if spoofing else "請檢查郵件真實性"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))