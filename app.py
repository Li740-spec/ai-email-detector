from flask import Flask, request, jsonify
from flask_cors import CORS  # <--- 核心：處理連線問題
import joblib
import os
import pandas as pd

app = Flask(__name__)
# 允許所有來源連線，解決「連線失敗」問題
CORS(app, resources={r"/*": {"origins": "*"}})

# 載入 AI 模型
try:
    clf = joblib.load('models/clf_zh.joblib')
    tfidf = joblib.load('models/tfidf_zh.joblib')
    print("模型載入成功！")
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
        if not text:
            return jsonify({'label': 'empty'})
        
        # AI 預測邏輯
        vec = tfidf.transform([text])
        label = clf.predict(vec)[0]
        return jsonify({'label': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        text = data.get('text', '')
        label = data.get('label', '')
        
        # 存入 CSV
        df = pd.DataFrame([[text, label]], columns=['Email Text', 'Email Type'])
        df.to_csv('feedback.csv', mode='a', index=False, header=not os.path.exists('feedback.csv'))
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)