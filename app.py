import os, joblib, re
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# 徹底開放 CORS 權限
CORS(app, resources={r"/*": {"origins": "*"}})

# 載入預訓練模型 (加上 try-except 防止模型遺失導致伺服器崩潰)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    clf = joblib.load(os.path.join(BASE_DIR, 'clf_zh.joblib'))
    tfidf = joblib.load(os.path.join(BASE_DIR, 'tfidf_zh.joblib'))
    print("✅ AI 偵測引擎載入成功")
except Exception as e:
    clf = None
    tfidf = None
    print(f"❌ 模型載入失敗: {e}")

# --- 新增：根目錄路由，防止出現 Not Found ---
@app.route('/')
def home():
    return "AI Email Detector Expert System is Running! (Endpoint: /predict)"

# --- 進階聰明邏輯：意圖偵測模式 ---
INTENT_PATTERNS = {
    "unauthorized_access": r"(登入|存取|訪問).*(異常|未知|設備|位置|裝置)", 
    "account_threat": r"(封鎖|停用|鎖定|刪除|限制).*(帳戶|帳號|功能)",
    "urgent_action": r"(立即|儘快|24小時|期限).*(驗證|更新|點擊|登入)"
}

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS': 
        return '', 200
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not clf or not tfidf:
            return jsonify({'error': '模型未就緒'}), 500

        # 1. 執行 AI 模型基礎預測
        vec = tfidf.transform([text])
        ai_prob = float(clf.predict_proba(vec)[0][1])

        # 2. 意圖推理 (Intent Reasoning)
        reasoning_score = 0
        detected_intents = []
        for intent, pattern in INTENT_PATTERNS.items():
            if re.search(pattern, text):
                reasoning_score += 0.15
                detected_intents.append(intent)

        # 3. 處理連結安全性 (結構分析)
        if re.search(r'http://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text):
            reasoning_score += 0.2

        # 4. 最終整合與判定
        final_prob = min(ai_prob + reasoning_score, 1.0)
        
        # 關鍵字過濾：排除長度小於 2 的字，並排除常見雜訊
        feature_names = tfidf.get_feature_names_out()
        weights = vec.toarray()[0]
        top_indices = weights.argsort()[::-1]
        
        # 排除清單
        stop_words = ['您的', '我們', '您可以', 'og', 'html', 'div']
        keywords = []
        for i in top_indices:
            word = str(feature_names[i])
            if len(word) > 1 and word not in stop_words and weights[i] > 0:
                keywords.append(word)
            if len(keywords) >= 3: break

        return jsonify({
            'label': 'phishing' if final_prob >= 0.5 else 'safe',
            'phish_prob': round(final_prob * 100, 1),
            'keywords': keywords,
            'reasoning_intents': detected_intents
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# --- 針對 Render 的啟動設定 ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)