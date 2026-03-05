import csv
import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 載入模型函數
def load_models():
    try:
        clf = joblib.load('models/clf_zh.joblib')
        vec = joblib.load('models/tfidf_zh.joblib')
        return clf, vec
    except:
        return None, None

@app.route('/')
def home():
    return "AI Server is running!"

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    content = data.get('content', '')
    clf, vec = load_models()
    
    if not clf or not vec:
        return jsonify({"error": "模型尚未訓練，請先執行 train_multilang.py"}), 500
    
    # 進行預測
    X = vec.transform([content])
    prob = round(clf.predict_proba(X)[0][1] * 100, 2)
    prediction = "Phishing Email" if prob > 50 else "Safe Email"
    
    print(f"--- 偵測請求 ---")
    print(f"內容節錄: {content[:30]}...")
    print(f"結果: {prediction} ({prob}%)")
    
    return jsonify({"prediction": prediction, "probability": prob})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    content = data.get('content', '')
    label = data.get('label', '')
    
    # 存入 feedback.csv (錯題本)
    file_exists = os.path.isfile('feedback.csv')
    with open('feedback.csv', mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Email Text', 'Email Type'])
        writer.writerow([content, label])
    
    print(f"📥 已存入糾正樣本：這封信其實是 {label}")
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)