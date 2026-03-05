from flask import Flask, request, jsonify
from flask_cors import CORS  # 必須安裝 pip install flask-cors
import joblib
import os
import pandas as pd
import numpy as np

app = Flask(__name__)
# 終極 CORS 設定：允許所有來源、所有標頭
CORS(app, resources={r"/*": {"origins": "*"}})

# 載入模型 (請確認 models 資料夾路徑正確)
try:
    clf = joblib.load('clf_zh.joblib')
    tfidf = joblib.load('tfidf_zh.joblib')
    print("AI Model Loaded Successfully!")
except Exception as e:
    print(f"Model Load Error: {e}")

@app.route('/')
def home():
    return "AI Email Detector Server is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        if not text or len(text) < 2:
            return jsonify({'label': 'safe', 'phish_prob': 0, 'keywords': []})
        
        vec = tfidf.transform([text])
        label = clf.predict(vec)[0]
        probs = clf.predict_proba(vec)[0] # [安全, 釣魚]
        phish_prob = round(probs[1] * 100, 1)
        
        # 抓取影響最大的詞 (可解釋性)
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
        print(f"Predict Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        df = pd.DataFrame([[data.get('text'), data.get('label')]], columns=['Email Text', 'Email Type'])
        df.to_csv('feedback.csv', mode='a', index=False, header=not os.path.exists('feedback.csv'))
        return jsonify({'status': 'success'})
    except:
        return jsonify({'status': 'error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
