import csv
import os
boost_data = [
    ("【國泰世華】您的網銀密碼已過期，請點擊連結重置：http://cathay-bk.com", "Phishing Email"),
    ("您的 Apple ID 存在安全風險，請立即驗證身份", "Phishing Email"),
    ("Netflix：您的帳戶付款失敗，請點擊此處更新信用卡資訊", "Phishing Email"),
    ("您的 Google 帳戶偵測到異常登入，請確認是否為本人", "Phishing Email"),
    ("蝦皮購物：您有未領取的優惠券，點擊連結領取：http://shopee-gift.top", "Phishing Email"),
    ("您的電費已逾期，請於24小時內繳清，否則將停止供電", "Phishing Email"),
    ("包裹投遞失敗，請支付 50 元手續費重新投遞", "Phishing Email")
]
expanded = []
for _ in range(50):
    for text, label in boost_data:
        expanded.append([text, label])
with open('user_feedback.csv', 'a', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(expanded)
print("✅ 成功注入 350 筆強化教材！")
