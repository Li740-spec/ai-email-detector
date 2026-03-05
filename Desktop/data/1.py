import sys, os
print("Python executable:", sys.executable)
try:
    import pandas as pd
    print("pandas version:", pd.__version__)
except Exception as e:
    print("pandas import error:", e)

try:
    import torch
    print("torch version:", torch.__version__)
except Exception as e:
    print("torch import error:", e)

try:
    import transformers
    print("transformers version:", transformers.__version__)
except Exception as e:
    print("transformers import error:", e)

try:
    import sklearn
    print("sklearn version:", sklearn.__version__)
except Exception as e:
    print("sklearn import error:", e)

import matplotlib
print("matplotlib version:", matplotlib.__version__)

# 列出 data 資料夾內容（若沒有則提示）
data_dir = os.path.join(os.path.dirname(__file__), "data")
if os.path.exists(data_dir) and os.path.isdir(data_dir):
    print("\nFiles in data/:")
    for f in os.listdir(data_dir):
        print(" -", f)
    # 嘗試讀取第一個 csv（若存在）
    csvs = [f for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
    if csvs:
        fp = os.path.join(data_dir, csvs[0])
        try:
            df = pd.read_csv(fp, nrows=5)
            print(f"\nPreview of {csvs[0]} (first 5 rows):")
            print(df.head())
        except Exception as e:
            print("Error reading CSV:", e)
else:
    print("\nNo data/ folder found in project dir. Create ~/Desktop/phishing_agent/data and put CSV files there if you have them.")
