import os
import joblib
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# 徹底開放 CORS 權限，確保插件連線不被擋
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 載入 AI 模型與向量化工具
try:
    clf = joblib.load(os.path.join(BASE_DIR, 'clf_zh.joblib'))
    tfidf = joblib.load(os.path.join(BASE_DIR, 'tfidf_zh.joblib'))
    print("✅ AI 偵測引擎載入成功")
except Exception as e:
    clf = None
    print(f"❌ 模型載入失敗: {e}")

@app.route('/')
def home():
    return "AI Email Detector Expert System is Running!"

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not clf:
            return jsonify({'error': 'AI 模型尚未就緒'}), 500
        if not text or len(text.strip()) < 5:
            return jsonify({'label': 'safe', 'phish_prob': 0.0, 'keywords': []})

        # --- 階段 1: AI 語意分析 ---
        vec = tfidf.transform([text])
        raw_prob = float(clf.predict_proba(vec)[0][1])  # 取得釣魚的機率 (0~1)
        
        # --- 階段 2: 規則引擎加成 (讓它變聰明的關鍵) ---
        boost_score = 0.0
        
        # 1. 偵測不安全連結 (http 而非 https)
        # 釣魚信常使用 http 網址或奇怪的 IP 連結
        if re.search(r'http://', text):
            boost_score += 0.25
            
        # 2. 偵測高度可疑關鍵字 (針對社交工程誘騙)
        danger_words = [
            '俄羅斯', '莫斯科', '嘗試登入', '暫時鎖定', 
            '立即驗證', '更新密碼', '永久停用', '中獎',
            '點擊下方', '異常活動', '驗證身份'
        ]
        
        hit_words = []
        for word in danger_words:
            if word in text:
                boost_score += 0.08
                hit_words.append(word)

        # 3. 計算最終機率 (原始 AI 分數 + 規則加成)
        # 這樣就算 AI 被官方語氣騙了，也會因為看到「俄羅斯」或「http」而警覺
        final_prob = min(raw_prob + boost_score, 1.0)
        
        # --- 階段 3: 最終判定 ---
        # 降低門檻，只要 40% 以上就標記為釣魚，寧可錯抓不放過
        final_label = 'phishing' if final_prob > 0.4 else 'safe'

        # 提取 TF-IDF 權重最高的詞作為特徵
        feature_names = tfidf.get_feature_names_out()
        weights = vec.toarray()[0]
        top_indices = weights.argsort()[-3:][::-1]
        keywords = [str(feature_names[i]) for i in top_indices if weights[i] > 0]

        return jsonify({
            'label': str(final_label),
            'phish_prob': float(round(final_prob * 100, 1)),
            'keywords': keywords,
            'analysis': {
                'ai_score': round(raw_prob, 2),
                'rule_boost': round(boost_score, 2),
                'detected_threats': hit_words
            }
        })
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 針對 Render 的部署設定
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)