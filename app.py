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

    # 1. 檢查惡意頂級域名 (TLDs) - 加重權重至 0.55
    malicious_tlds = [r'\.top', r'\.xyz', r'\.cc', r'\.info', r'\.pw', r'\.icu']
    for tld in malicious_tlds:
        if re.search(tld, text, re.IGNORECASE):
            bonus += 0.55 
            reasons.append("Suspicious Domain (.top/.xyz)")
            break

    # 2. 檢查品牌劫持 (Brand Squatting) - 加重權重至 0.50
    if re.search(r'google|gmail|account-verify|shopee|bank', text, re.IGNORECASE):
        if not re.search(r'google\.com|gmail\.com|shopee\.tw', text, re.IGNORECASE):
            bonus += 0.50
            reasons.append("Brand Squatting (Fake Official link)")

    # 3. 檢查急迫性詞組 (Urgency Cues)
    urgency_patterns = [r'立即', r'24小時', r'最後', r'異常', r'驗證碼']
    hit_count = sum(1 for p in urgency_patterns if re.search(p, text))
    if hit_count >= 2:
        bonus += 0.15
        reasons.append("High Urgency Language")

    return bonus, reasons

# --- 路由設定 ---

@app.route('/')
def home(): 
    return jsonify({"status": "Online", "engine_version": "2.1-Hybrid-Optimized"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        text = data.get('text', '')
        text_lower = text.lower()

        # --- [優化 1：白名單攔截器] 防止誤殺好信 ---
        # 只要郵件包含這些官方域名，直接給低分回傳
        safe_domains = ['nkust.edu.tw', 'gmail.com', 'canva.com', 'google.com', 'edu.tw', 'github.com']
        if any(domain in text_lower for domain in safe_domains):
            return jsonify({
                'phish_prob': 5.0, # 顯示極低風險
                'threat_category': ["Official / Safe Domain"],
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
        
        # --- [優化 2：保底加權邏輯] 確保壞信亮紅燈 ---
        # 如果觸發了專家規則，保底分數至少 0.75 起跳 (對應紅色警戒)
        if bonus_score > 0:
            final_prob = max(0.75, min(ai_prob + bonus_score, 1.0))
        else:
            final_prob = ai_prob
        
        # 3. 威脅類別整合
        base_cats = []
        if re.search(r'登入|驗證|密碼', text): base_cats.append("Credential Phishing")
        if re.search(r'中獎|免費|領取', text): base_cats.append("Lure/Baiting")
        
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