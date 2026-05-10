import os
import math
import joblib
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

        # 2. AI 原始預測 (0~1 之間)
        vec = tfidf.transform([text])
        ai_prob = float(clf.predict_proba(vec)[0][1])

        # 3. 邏輯校準：判斷是否為校內來源
        is_nkust = "nkust.edu.tw" in sender.lower() or "nkust.edu.tw" in text.lower()
        
        # 4. 關鍵字強效偵測 (防止 AI 漏看明顯的威脅)
        danger_keywords = ['登入', '驗證', '密碼', '停用', '點擊', '更新帳戶', '異常']
        hit_keywords = [word for word in danger_keywords if word in text]
        
        # --- 核心決策引擎 ---
        if is_nkust:
            if len(hit_keywords) >= 2 or ai_prob > 0.7:
                # 即使掛著學校網域，只要內容太像釣魚，依然判定為中高風險
                final_prob = max(ai_prob, 0.45)
                label = "校內網域 (但內容疑似異常)"
            else:
                # 真正的校內公務信：大幅降分
                final_prob = (ai_prob * 0.2) + 0.015
                label = "校內公務郵件 (安全)"
        else:
            # 外部郵件：完全依照 AI 判定，並對關鍵字進行加權
            bonus = 0.15 if hit_keywords else 0.0
            final_prob = min(ai_prob + bonus, 0.99)
            label = "外部郵件偵測"

        # 5. Sigmoid 曲線優化 (讓 1%~99% 的分佈更符合直覺)
        # 確保不會死板的只有 1%，而是在 1.5%~3% 之間有細微跳動
        final_prob = 1 / (1 + math.exp(-8 * (final_prob - 0.4)))
        
        # 最終數值校正：保底 1.5%，上限 98.5%
        display_prob = max(min(final_prob, 0.985), 0.015)

        return jsonify({
            "phish_prob": round(display_prob * 100, 1),
            "probability": f"{display_prob * 100:.1f}%",
            "status": "危險" if display_prob > 0.5 else "安全",
            "label": label,
            "is_official": is_nkust,
            "engine": "NKUST-Dual-Core v3.6"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({"status": "Online", "version": "3.6-Final-Precision"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
