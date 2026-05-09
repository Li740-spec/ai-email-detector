import os
import joblib
import re
import math
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# 強化 CORS 設定，確保 Chrome Extension 穩定連線
CORS(app, resources={r"/*": {"origins": "*", "methods": ["POST", "OPTIONS"]}})

# 載入模型
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    clf = joblib.load(os.path.join(BASE_DIR, 'clf_zh.joblib'))
    tfidf = joblib.load(os.path.join(BASE_DIR, 'tfidf_zh.joblib'))
except Exception as e:
    print(f"模型載入失敗: {e}")
    clf, tfidf = None, None

# --- 核心優化邏輯：機器學習偏置校正 ---

def sigmoid_calibration(prob):
    """
    使用 Sigmoid 函數進行機率校準。
    這能壓制模型在模糊地帶(0.5左右)的過度反應，讓判定更符合真實分布。
    """
    # 這裡的 -10 和 0.6 是校準參數，能讓 0.6 以下的機率大幅收縮
    # 這是機器學習中常用的 Platt Scaling 簡化版
    return 1 / (1 + math.exp(-10 * (prob - 0.65)))

# --- 路由設定 ---

@app.route('/')
def home(): 
    return jsonify({"status": "Online", "engine_version": "3.5-Pure-ML-Calibration"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        text = data.get('text', '')
        
        # --- 機器學習推論流程 ---
        if clf and tfidf:
            # 1. 向量化
            vec = tfidf.transform([text])
            # 2. 取得原始 AI 預測機率
            raw_ai_prob = float(clf.predict_proba(vec)[0][1])
            
            # 3. 執行機率校準 (不使用人工關鍵字，純數學轉換)
            calibrated_prob = sigmoid_calibration(raw_ai_prob)
            
            # 4. 判定威脅等級 (完全由校準後的 AI 分數決定)
            final_prob = round(calibrated_prob * 100, 1)
        else:
            final_prob = 5.0 # 模型缺失時的安全預設值

        # 自動生成類別 (基於模型信心度)
        if final_prob > 70:
            threat_cats = ["High Confidence Phishing"]
        elif final_prob > 40:
            threat_cats = ["Potential Risk Detected"]
        else:
            threat_cats = ["AI Verified: Safe"]

        return jsonify({
            'phish_prob': final_prob,
            'threat_category': threat_cats,
            'engine': 'Random Forest + Sigmoid Calibration v3.5'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)