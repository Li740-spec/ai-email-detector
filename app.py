import os
import math
import joblib
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 1. 載入模型與向量化工具
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
        # 相容多種前端欄位名稱
        text = data.get('content') or data.get('text', '')
        sender = data.get('sender', '')

        if not model_loaded:
            return jsonify({"error": "Model not loaded"}), 500

        # 2. AI 原始權重預測
        vec = tfidf.transform([text])
        ai_prob = float(clf.predict_proba(vec)[0][1])

        # 3. 診斷標籤邏輯
        threat_categories = []
        is_suspicious_domain = False
        
        # 偵測惡意頂級域名
        if re.search(r'\.(top|xyz|cc|pw|icu|info|tk)', text, re.IGNORECASE):
            threat_categories.append("Suspicious Domain (.top/.xyz)")
            is_suspicious_domain = True
        
        # 偵測急迫性語氣
        if re.search(r'立即|最後機會|24小時|緊急|儘速', text):
            threat_categories.append("High Urgency Language")
        
        # 偵測憑證竊取關鍵字
        if re.search(r'登入|驗證|密碼|更新帳戶|點擊連結|身分確認', text):
            threat_categories.append("Credential Phishing")

        # 4. 權重與偏移計算
        is_nkust = "nkust.edu.tw" in sender.lower() or "nkust.edu.tw" in text.lower()
        
        # 基礎分數疊加
        bonus = len(threat_categories) * 0.12
        if is_suspicious_domain:
            bonus += 0.18
            
        final_score = ai_prob + bonus

        # 5. v3.9 平滑 Sigmoid 曲線處理 (解決 98.5% 太高的問題)
        # 降低斜率 (-6) 並調高中位數 (0.48)，讓數值變化更細膩
        display_prob = 1 / (1 + math.exp(-6 * (final_score - 0.48)))
        
        # 真實感偏移處理：避免出現過於絕對的數字
        if display_prob > 0.90:
            # 讓高分信件落在 91%~95% 之間跳動
            display_prob = 0.88 + (display_prob * 0.05)
        elif display_prob < 0.10:
            # 讓低分信件落在 2%~5% 之間跳動
            display_prob = 0.02 + (display_prob * 0.2)

        # 6. 校內信特殊優待 (僅限無威脅標籤時)
        if is_nkust and not threat_categories:
            display_prob = display_prob * 0.15
            status_label = "安全：校內公務信件"
        else:
            status_label = "高風險：偵測到釣魚威脅" if display_prob > 0.5 else "安全：外部郵件"

        # 確保保底數值
        display_prob = max(min(display_prob, 0.96), 0.018)

        # 7. 最終回傳：對齊前端所有預期欄位
        return jsonify({
            "phish_prob": round(display_prob * 100, 1),
            "probability": f"{display_prob * 100:.1f}%",
            "status": "危險" if display_prob > 0.5 else "安全",
            "threat_category": threat_categories if threat_categories else ["General Analysis"],
            "label": status_label,
            "is_official": is_nkust,
            "engine": "Precision Engine v3.9"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({
        "status": "Online",
        "version": "3.9-Precision-Smooth",
        "model_loaded": model_loaded
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
