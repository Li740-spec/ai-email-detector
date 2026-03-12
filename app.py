import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# 徹底開放權限，防止插件連線失敗
CORS(app, resources={r"/*": {"origins": "*"}})

# 設定模型路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 載入模型與向量化工具
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
        text = data.get('text', '測試郵件內容')

        if not clf:
            return jsonify({'error': '模型未就緒'}), 500

        # 預測
        vec = tfidf.transform([text])
        label = clf.predict(vec)[0]
        prob = clf.predict_proba(vec)[0][1] # 釣魚機率

        # 關鍵字提取 (選權重最高的前三個)
        feature_names = tfidf.get_feature_names_out()
        weights = vec.toarray()[0]
        top_indices = weights.argsort()[-3:][::-1]
        keywords = [feature_names[i] for i in top_indices if weights[i] > 0]

        return jsonify({
            'label': label,
            'phish_prob': round(prob * 100, 1),
            'keywords': keywords
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)