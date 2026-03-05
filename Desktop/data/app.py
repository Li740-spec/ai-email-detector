from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd
import numpy as np

app = Flask(__name__)
# 允許所有來源連線，解決插件跨網域問題
CORS(app, resources={r"/*": {"origins": "*"}})

# 載入 AI 模型
try:
    clf = joblib.load('models/clf_zh.joblib')
    tfidf = joblib.load('models/tfidf_zh.joblib')
    print("AI 模型載入成功！")
except Exception as e:
    print(f"模型載入失敗: {e}")

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
        
        # 1. 向量化
        vec = tfidf.transform([text])
        
        # 2. 預測類別與機率 (信心指數)
        label = clf.predict(vec)[0]
        probs = clf.predict_proba(vec)[0] # 得到 [安全機率, 釣魚機率]
        phish_prob = round(probs[1] * 100, 1) # 取得釣魚機率百分比
        
        # 3. 可解釋性：找出文本中特徵分數最高的詞
        feature_names = tfidf.get_feature_names_out()
        words_weights = vec.toarray()[0]
        # 取得分數最高的前 3 個詞
        top_indices = words_weights.argsort()[-3:][::-1]
        keywords = [feature_names[i] for i in top_indices if words_weights[i] > 0]

        return jsonify({
            'label': label,
            'phish_prob': phish_prob,
            'keywords': keywords
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        text = data.get('text', '')
        label = data.get('label', '')
        # 儲存回饋數據，實現 MLOps 數據閉環
        df = pd.DataFrame([[text, label]], columns=['Email Text', 'Email Type'])
        df.to_csv('feedback.csv', mode='a', index=False, header=not os.path.exists('feedback.csv'))
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)