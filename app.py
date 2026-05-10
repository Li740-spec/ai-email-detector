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

        # 3. 診斷標籤邏輯
        threat_categories = []
        is_suspicious_domain = False
        
        if re.search(r'\.(top|xyz|cc|pw|icu|info|tk)', text, re.IGNORECASE):
            threat_categories.append("Suspicious Domain (.top/.xyz)")
            is_suspicious_domain = True
        
        if re.search(r'立即|最後機會|24小時|緊急', text):
            threat_categories.append("High Urgency Language")
        
        if re.search(r'登入|驗證|密碼|更新帳戶|點擊連結', text):
            threat_categories.append("Credential Phishing")

        # 4. 核心決策引擎：拉開差距
        is_nkust = "nkust.edu.tw" in sender.lower() or "nkust.edu.tw" in text.lower()
        
        # 基礎加權：每個威脅標籤提供 0.15 的基礎分
        bonus = len(threat_categories) * 0.15
        
        # 如果有惡意網域，再額外重罰 0.2
        if is_suspicious_domain:
            bonus += 0.2

        final_prob = ai_prob + bonus

        # --- 核心關鍵：強制閾值 ---
        # 如果標籤超過 2 個，或是 AI 覺得這封信真的很怪，強制拉到危險水位
        if len(threat_categories) >= 2 or (is_suspicious_domain and ai_prob > 0.3):
            final_prob = max(final_prob, 0.6) # 直接保底 60%

        # 5. Sigmoid 優化 (讓邊界更陡峭，危險的直接衝上去)
        # 我們將中位數從 0.45 降到 0.4，讓判定變嚴格
        display_prob = 1 / (1 + math.exp(-12 * (final_prob - 0.4)))
        
        # 如果是校內信且完全沒有標籤，才給予壓分
        if is_nkust and not threat_categories:
            display_prob = display_prob * 0.1
            status_text = "安全：校內公務信件"
        else:
            status_text = "高風險：偵測到釣魚威脅" if display_prob > 0.5 else "安全：外部郵件"

        # 數值修正：確保不為 0
        display_prob = max(min(display_prob, 0.985), 0.012)

        return jsonify({
            "phish_prob": round(display_prob * 100, 1),
            "probability": f"{display_prob * 100:.1f}%",
            "status": "危險" if display_prob > 0.5 else "安全",
            "threat_category": threat_categories if threat_categories else ["General Analysis"],
            "label": status_text,
            "is_official": is_nkust
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({"status": "Online", "version": "3.8-Strict-Mode"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
