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

    # 1. 檢查惡意頂級域名 (TLDs)
    malicious_tlds = [r'\.top', r'\.xyz', r'\.cc', r'\.info', r'\.pw', r'\.icu']
    for tld in malicious_tlds:
        if re.search(tld, text, re.IGNORECASE):
            bonus += 0.25
            reasons.append("Suspicious Domain (.top/.xyz)")
            break

    # 2. 檢查品牌劫持 (Brand Squatting)
    # 如果網址包含 google/gmail 但不是官方來源
    if re.search(r'google|gmail|account-verify', text, re.IGNORECASE):
        if not re.search(r'google\.com|gmail\.com', text, re.IGNORECASE):
            bonus += 0.30
            reasons.append("Brand Squatting (Fake Google link)")

    # 3. 檢查急迫性詞組 (Urgency Cues)
    urgency_patterns = [r'立即', r'24小時', r'永久停用', r'異常登入', r'最後機會']
    hit_count = sum(1 for p in urgency_patterns if re.search(p, text))
    if hit_count >= 2:
        bonus += (hit_count * 0.05)
        reasons.append("High Urgency Language")

    return bonus, reasons

# --- 路由設定 ---

@app.route('/')
def home(): 
    return jsonify({"status": "Online", "engine_version": "2.1-Hybrid"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        text = data.get('text', '')
        
        # 1. 基礎 AI 預測 (隨機森林)
        if clf and tfidf:
            vec = tfidf.transform([text])
            ai_prob = float(clf.predict_proba(vec)[0][1])
        else:
            ai_prob = 0.5 # 模型未載入時的預設值

        # 2. 引入啟發式加權 (Heuristic Bonus)
        bonus_score, extra_cats = get_heuristic_bonus(text)
        
        # 最終分數計算 (加權總和，最高 1.0)
        final_prob = min(ai_prob + bonus_score, 1.0)
        
        # 3. 威脅類別整合
        base_cats = []
        if re.search(r'登入|驗證|密碼', text): base_cats.append("Credential Phishing")
        if re.search(r'中獎|免費|領取', text): base_cats.append("Lure/Baiting")
        
        # 合併 AI 判斷與規則判斷的類別
        all_categories = list(set(base_cats + extra_cats))

        return jsonify({
            'phish_prob': round(final_prob * 100, 1),
            'threat_category': all_categories if all_categories else ["General Analysis"],
            'engine': 'Random Forest + Heuristic v2.1'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Render 專用端口設定
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)