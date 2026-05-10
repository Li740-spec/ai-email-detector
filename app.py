import os
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
# 強化 CORS 設定，確保擴充功能穩定連線
CORS(app)

# 1. 自動偵測路徑並載入模型
base_path = os.path.dirname(os.path.abspath(__file__))
try:
    model = joblib.load(os.path.join(base_path, 'clf_zh.joblib'))
    tfidf = joblib.load(os.path.join(base_path, 'tfidf_zh.joblib'))
    model_loaded = True
except Exception as e:
    print(f"模型載入失敗: {e}")
    model_loaded = False

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # 相容舊版前端可能傳送的 'text' 或新版的 'content'
        email_content = data.get('content') or data.get('text', '')
        sender = data.get('sender', '')
        
        if not model_loaded:
            return jsonify({"error": "Server model not loaded"}), 500

        # 2. 執行 AI 原始預測
        email_tfidf = tfidf.transform([email_content])
        raw_prob = model.predict_proba(email_tfidf)[0][1]
        
        # 3. Sigmoid 數學校準 (修正模型邊界)
        calibrated_prob = 1 / (1 + math.exp(-10 * (raw_prob - 0.65)))
        
        # 4. 針對校內網域進行「信任權重偏移」
        is_nkust = "nkust.edu.tw" in sender.lower() or "nkust.edu.tw" in email_content.lower()
        
        if is_nkust:
            # 校內信：風險權重減半並加保底
            final_prob = (calibrated_prob * 0.5) + 0.02
        else:
            final_prob = calibrated_prob

        # 5. 最終保底機制：確保不會顯示 0.0%
        final_prob = max(final_prob, 0.012)

        status = "危險" if final_prob > 0.5 else "安全"
        
        # 6. 回傳前端認得的欄位 (phish_prob)
        return jsonify({
            "phish_prob": round(final_prob * 100, 1), # 前端顯示用的數字
            "probability": f"{final_prob * 100:.1f}%",
            "status": status,
            "label": "AI 深度偵測完成 (v3.5)",
            "is_official": is_nkust,
            "engine": "Precision Engine v3.5"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({
        "status": "Online",
        "model_status": "Loaded" if model_loaded else "Error",
        "version": "3.5-Precision-Final"
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
