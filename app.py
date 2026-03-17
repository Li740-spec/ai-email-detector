import os, joblib, re
from flask import Flask, request, jsonify
from flask_cors import CORS
from difflib import SequenceMatcher

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    clf = joblib.load(os.path.join(BASE_DIR, 'clf_zh.joblib'))
    tfidf = joblib.load(os.path.join(BASE_DIR, 'tfidf_zh.joblib'))
except Exception as e:
    clf, tfidf = None, None

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
        
        # 邏輯運算
        urls = re.findall(r'https?://([a-zA-Z0-9.-]+)', text)
        spoofing = False
        for url in urls:
            domain = url.split('.')[0].lower()
            for brand in ["google", "facebook", "apple", "microsoft"]:
                if 0.75 < get_similarity(brand, domain) < 1.0: spoofing = True

        vec = tfidf.transform([text]) if tfidf else None
        ai_prob = float(clf.predict_proba(vec)[0][1]) if clf else 0.5

        cats = []
        if re.search(r'登入|驗證|密碼', text): cats.append("Credential Phishing")
        if re.search(r'立即|緊急|失效', text): cats.append("Social Engineering")

        final_prob = min(ai_prob + (0.2 if cats else 0) + (0.4 if spoofing else 0), 1.0)

        return jsonify({
            'label': 'phishing' if final_prob >= 0.5 else 'safe',
            'phish_prob': round(final_prob * 100, 1),
            'threat_category': cats if cats else ["General Analysis"],
            'spoofing_alert': spoofing
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))