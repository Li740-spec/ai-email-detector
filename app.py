import os
import joblib
import re
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

# --- 核心優化邏輯：混合偵測引擎 ---

def get_heuristic_bonus(text):
    """
    啟發式規則加權：針對 AI 模型容易漏看的資安特徵進行人工加權
    """
    bonus = 0.0
    reasons = []
    text_lower = text.lower()

    # 1. 檢查惡意頂級域名 (TLDs)
    malicious_tlds = [r'\.top', r'\.xyz', r'\.cc', r'\.info', r'\.pw', r'\.icu']
    for tld in malicious_tlds:
        if re.search(tld, text_lower):
            bonus += 0.55 
            reasons.append("Suspicious Domain (.top/.xyz)")
            break

    # 2. 檢查品牌劫持 (Brand Squatting) 
    # 排除校內郵件 (nkust.edu.tw) 常見的簽名檔誤判
    if re.search(r'google|gmail|shopee|bank|verification', text_lower):
        if not re.search(r'google\.com|gmail\.com|shopee\.tw|nkust\.edu\.tw', text_lower):
            bonus += 0.50
            reasons.append("Brand Squatting (Fake Official link)")

    # 3. 檢查急迫性詞組 (Urgency Cues)
    urgency_patterns = [r'立即', r'24小時', r'最後', r'異常', r'驗證碼']
    hit_count = sum(1 for p in urgency_patterns if re.search(p, text_lower))
    if hit_count >= 2:
        bonus += 0.15
        reasons.append("High Urgency Language")

    return bonus, reasons

# --- 路由設定 ---

@app.route('/')
def home(): 
    return jsonify({"status": "Online", "engine_version": "2.1-Hybrid-Final"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        text = data.get('text', '')
        text_lower = text.lower()

        # --- [優化：白名單攔截器] 解決校內郵件誤判 ---
        # 加入學校網域、常用官方網域，以及助教信箱關鍵字 (vicky923)
        safe_keywords = ['nkust.edu.tw', 'vicky923', 'gmail.com', 'canva.com', 'google.com', 'edu.tw']
        if any(keyword in text_lower for keyword in safe_keywords):
            return jsonify({
                'phish_prob': 5.0, # 綠燈安全
                'threat_category': ["Official / University Mail"],
                'engine': 'White-list Bypass'
            })

        # --- [核心偵測流程] ---
        # 1. 基礎 AI 預測 (隨機森林)
        if clf and tfidf:
            vec = tfidf.transform([text])
            ai_prob = float(clf.predict_proba(vec)[0][1])
        else:
            ai_prob = 0.5

        # 2. 引入啟發式加權 (Heuristic Bonus)
        bonus_score, extra_cats = get_heuristic_bonus(text)
        
        # 3. 保底加權邏輯
        if bonus_score > 0:
            final_prob = max(0.75, min(ai_prob + bonus_score, 1.0))
        else:
            final_prob = ai_prob
        
        base_cats = []
        if re.search(r'登入|驗證|密碼', text_lower): base_cats.append("Credential Phishing")
        if re.search(r'中獎|免費|領取', text_lower): base_cats.append("Lure/Baiting")
        
        all_categories = list(set(base_cats + extra_cats))

        return jsonify({
            'phish_prob': round(final_prob * 100, 1),
            'threat_category': all_categories if all_categories else ["General Analysis"],
            'engine': 'Random Forest + Heuristic v2.1-Hybrid'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)