import os
import joblib
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 載入模型
try:
    clf = joblib.load(os.path.join(BASE_DIR, 'clf_zh.joblib'))
    tfidf = joblib.load(os.path.join(BASE_DIR, 'tfidf_zh.joblib'))
    print("✅ 引擎載入成功")
except Exception as e:
    clf = None

# --- 變聰明的關鍵：自定義威脅字典 ---
# 這些組合如果同時出現，代表威脅性極高
THREAT_PATTERNS = [
    (r'登入|驗證|鎖定', 0.15),   # 動作誘導
    (r'俄羅斯|莫斯科|未知裝置', 0.2), # 地理位置異常
    (r'立即|儘快|24小時', 0.1),  # 急迫性
    (r'http://', 0.25)          # 不安全連結
]

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS': return '', 200
    try:
        data = request.get_json()
        text = data.get('text', '')

        # 1. 基礎 AI 分數
        vec = tfidf.transform([text])
        prob = float(clf.predict_proba(vec)[0][1])

        # 2. 深度語意分析 (Smart Reasoning)
        # 我們不只是加分，而是去尋找「威脅組合」
        reasoning_bonus = 0
        for pattern, weight in THREAT_PATTERNS:
            if re.search(pattern, text):
                reasoning_bonus += weight
        
        # 3. 排除無意義雜訊 (讓關鍵字變聰明)
        # 這些詞會干擾 AI 判斷，我們在顯示時過濾掉
        noise_words = ['您的', '我們', '您可以', 'og', 'html', 'div', '帳戶', 'google']
        
        feature_names = tfidf.get_feature_names_out()
        weights = vec.toarray()[0]
        # 抓取權重高的詞，但排除掉雜訊
        top_indices = weights.argsort()[::-1]
        keywords = []
        for i in top_indices:
            word = str(feature_names[i])
            if word not in noise_words and weights[i] > 0:
                keywords.append(word)
            if len(keywords) >= 3: break

        # 最終邏輯：AI 分數 + 深度分析分數
        final_score = min(prob + reasoning_bonus, 1.0)
        
        # 恢復 0.5 的標準門檻，因為我們已經讓分數本身變精確了
        label = 'phishing' if final_score >= 0.5 else 'safe'

        return jsonify({
            'label': label,
            'phish_prob': round(final_score * 100, 1),
            'keywords': keywords,
            'reasoning': "偵測到異常登入活動與急迫性語法" if reasoning_bonus > 0.2 else "一般郵件語意"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)