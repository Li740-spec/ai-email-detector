import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# 徹底開放 CORS，解決插件連線被擋的問題
CORS(app, resources={r"/*": {"origins": "*"}})

# 定義路徑，確保 Render 找得到模型
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 載入模型
try:
    clf = joblib.load(os.path.join(BASE_DIR, 'clf_zh.joblib'))
    tfidf = joblib.load(os.path.join(BASE_DIR, 'tfidf_zh.joblib'))
    print("✅ AI 模型載入成功！")
except Exception as e:
    clf = None
    tfidf = None
    print(f"❌ 模型載入失敗: {e}")

@app.route('/')
def home():
    status = "Ready" if clf else "Model Missing"
    return f"AI Email Detector Server is Running! (Status: {status})"

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # 處理瀏覽器預檢請求
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not clf or not tfidf:
            return jsonify({'error': 'Server model not ready'}), 500

        if not text:
            return jsonify({'label': 'safe', 'phish_prob': 0, 'keywords': []})

        # AI 預測邏輯
        vec = tfidf.transform([text])
        label = clf.predict(vec)[0]
        probs = clf.predict_proba(vec)[0]
        phish_prob = round(probs[1] * 100, 1)
        
        # 關鍵字提取
        feature_names = tfidf.get_feature_names_out()
        weights = vec.toarray()[0]
        top_indices = weights.argsort()[-3:][::-1]
        keywords = [feature_names[i] for i in top_indices if weights[i] > 0]

        return jsonify({
            'label': label,
            'phish_prob': phish_prob,
            'keywords': keywords
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 這裡必須配合 Render 的環境變數
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)