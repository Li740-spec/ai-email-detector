import csv
import os
safe_data = [
    ("關於下週一的會議，請大家準時參加並準備報告", "Safe Email"),
    ("您的電子發票已開立，金額為 150 元，祝您中獎", "Safe Email"),
    ("感謝您在 PChome 購買的商品，目前已安排出貨", "Safe Email"),
    ("外送訂單已送達，請到門口領取，祝您用餐愉快", "Safe Email"),
    ("這是您本月的薪資明細表，如有疑問請洽人資部", "Safe Email"),
    ("親愛的家長您好，這是本週的聯絡簿通知事項", "Safe Email")
]
expanded = []
for _ in range(60):
    for text, label in safe_data:
        expanded.append([text, label])
with open('user_feedback.csv', 'a', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(expanded)
print("✅ 成功注入 360 筆「安全」教材！")
