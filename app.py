import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# 徹底開放 CORS 權限
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 載入模型
try:
    clf = joblib.load(os.path.join(BASE_DIR, 'clf_zh.joblib'))
    tfidf = joblib.load(os.path.join(BASE_DIR, 'tfidf_zh.joblib'))
    print("✅ AI 模型載入成功")
except Exception as e:
    clf = None
    print(f"❌ 模型載入失敗: {e}")

@app.route('/')
def home():
    return "AI Email Detector Server is Running!"

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not clf:
            return jsonify({'error': '模型未就緒'}), 500
        if not text:
            return jsonify({'label': 'safe', 'phish_prob': 0.0, 'keywords': []})

        # AI 預測
        vec = tfidf.transform([text])
        label = clf.predict(vec)[0]
        prob = clf.predict_proba(vec)[0][1]

        # 關鍵字提取
        feature_names = tfidf.get_feature_names_out()
        weights = vec.toarray()[0]
        top_indices = weights.argsort()[-3:][::-1]
        # 確保所有變數都轉型為原生 Python 類型 (str, float)
        keywords = [str(feature_names[i]) for i in top_indices if weights[i] > 0]

        return jsonify({
            'label': str(label),
            'phish_prob': float(round(prob * 100, 1)),
            'keywords': keywords
        })
    except Exception as e:
        print(f"預測出錯: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)