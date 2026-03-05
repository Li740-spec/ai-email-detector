from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd

app = Flask(__name__)
CORS(app)  # 這行最重要，允許插件從不同網址存取

# 載入模型
try:
    clf = joblib.load('models/clf_zh.joblib')
    tfidf = joblib.load('models/tfidf_zh.joblib')
except:
    print("模型載入失敗，請確認 models 資料夾路徑。")

@app.route('/')
def home():
    return "AI Email Detector Server is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    # 預測邏輯
    vec = tfidf.transform([text])
    label = clf.predict(vec)[0]
    
    return jsonify({'label': label})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    text = data.get('text', '')
    label = data.get('label', '')
    
    # 存入 CSV
    df = pd.DataFrame([[text, label]], columns=['Email Text', 'Email Type'])
    df.to_csv('feedback.csv', mode='a', index=False, header=not os.path.exists('feedback.csv'))
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    # Render 會自動指定 PORT
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)