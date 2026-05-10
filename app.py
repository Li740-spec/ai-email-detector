import os
import math
import joblib
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 1. 載入模型
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    clf = joblib.load(os.path.join(BASE_DIR, 'clf_zh.joblib'))
    tfidf = joblib.load(os.path.join(BASE_DIR, 'tfidf_zh.joblib'))
    model_loaded = True
except Exception as e:
    print(f"模型載入失敗: {e}")
    model_loaded = False

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('content') or data.get('text', '')
        sender = data.get('sender', '')

        if not model_loaded:
            return jsonify({"error": "Model not loaded"}), 500

        # 2. AI 原始預測
        vec = tfidf.transform([text])
        ai_prob = float(clf.predict_proba(vec)[0][1])

        # 3. 診斷標籤邏輯 (就是你要找回來的那些標籤)
        threat_categories = []
        
        # 檢查惡意網域特徵
        if re.search(r'\.(top|xyz|cc|pw|icu)', text, re.IGNORECASE):
            threat_categories.append("Suspicious Domain (.top/.xyz)")
        
        # 檢查急迫性文字
        if re.search(r'立即|最後機會|24小時', text):
            threat_categories.append("High Urgency Language")
        
        # 檢查憑證竊取特徵
        if re.search(r'登入|驗證|密碼|更新帳戶', text):
            threat_categories.append("Credential Phishing")

        # 4. 核心決策與校準
        is_nkust = "nkust.edu.tw" in sender.lower() or "nkust.edu.tw" in text.lower()
        
        if is_nkust:
            # 校內信且沒有明顯威脅特徵，才給予降分
            if len(threat_categories) == 0:
                final_prob = (ai_prob * 0.2) + 0.015
                status_text = "安全：校內公務信件"
            else:
                final_prob = max(ai_prob, 0.42) # 有威脅特徵則維持中度風險
                status_text = "警示：校內來源但內容可疑"
        else:
            # 外部郵件，根據 AI 與標籤數量加成
            bonus = len(threat_categories) * 0.1
            final_prob = min(ai_prob + bonus, 0.99)
            status_text = "危險：偵測到釣魚威脅" if final_prob > 0.5 else "安全：外部郵件"

        # Sigmoid 平滑處理
        display_prob = 1 / (1 + math.exp(-8 * (final_prob - 0.45)))
        display_prob = max(min(display_prob, 0.985), 0.015)

        # 5. 回傳所有前端需要的欄位
        return jsonify({
            "phish_prob": round(display_prob * 100, 1), # 顯示圓圈百分比
            "probability": f"{display_prob * 100:.1f}%",
            "status": "危險" if display_prob > 0.5 else "安全",
            "threat_category": threat_categories if threat_categories else ["General Analysis"], # 這是你要的標籤！
            "label": status_text,
            "is_official": is_nkust
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({"status": "Online", "version": "3.7-Diagnosis-Restore"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
