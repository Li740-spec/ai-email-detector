import os, joblib, re
from flask import Flask, request, jsonify
from flask_cors import CORS
from difflib import SequenceMatcher

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 載入模型與向量化器
try:
    clf = joblib.load(os.path.join(BASE_DIR, 'clf_zh.joblib'))
    tfidf = joblib.load(os.path.join(BASE_DIR, 'tfidf_zh.joblib'))
except Exception as e:
    clf = None

# --- 專業偵測設定 ---
TRUSTED_BRANDS = ["google", "facebook", "apple", "microsoft", "amazon", "line", "netflix"]
MALICIOUS_DOMAINS = ["verify-g00gle.com", "login-microsoft.top", "account-update.xyz"]

def get_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

@app.route('/')
def home():
    return "Cybersecurity AI Engine v2.0 - Status: Online"

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS': return '', 200
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # A. 域名與連結掃描
        urls = re.findall(r'https?://([a-zA-Z0-9.-]+)', text)
        spoofing_detected = False
        on_blacklist = False
        for url in urls:
            if any(bad in url for bad in MALICIOUS_DOMAINS): on_blacklist = True
            for brand in TRUSTED_BRANDS:
                sim = get_similarity(brand, url.split('.')[0].lower())
                if 0.75 < sim < 1.0: spoofing_detected = True

        # B. AI 預測
        vec = tfidf.transform([text])
        ai_prob = float(clf.predict_proba(vec)[0][1]) if clf else 0.5

        # C. 意圖分類 (Threat Categorization)
        categories = []
        boost = 0
        if re.search(r'登入|驗證|密碼|身分', text):
            categories.append("Credential Phishing")
            boost += 0.2
        if re.search(r'立即|緊急|失效|限制', text):
            categories.append("Social Engineering")
            boost += 0.2
        if urls:
            categories.append("Malicious Link")
            boost += 0.1

        # D. 最終權重計算
        final_prob = ai_prob + boost
        if spoofing_detected: final_prob += 0.4
        if on_blacklist: final_prob = 1.0
        final_prob = min(final_prob, 1.0)

        # E. 關鍵字過濾 (去除雜訊)
        feature_names = tfidf.get_feature_names_out()
        weights = vec.toarray()[0]
        top_indices = weights.argsort()[::-1]
        stop_words = ['您的', '我們', '您可以', 'html', 'le', 'gle', 'title', 'meta']
        keywords = [str(feature_names[i]) for i in top_indices 
                    if len(str(feature_names[i])) > 1 and str(feature_names[i]) not in stop_words][:3]

        return jsonify({
            'label': 'phishing' if final_prob >= 0.5 else 'safe',
            'phish_prob': round(final_prob * 100, 1),
            'threat_category': categories if categories else ["General Pattern"],
            'keywords': keywords,
            'spoofing_alert': spoofing_detected,
            'advice': "🚩 高風險：偵測到域名偽裝！" if spoofing_detected else "請檢查連結真實性"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)