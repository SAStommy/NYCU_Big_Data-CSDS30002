import os
import urllib.request
from datetime import datetime, timedelta

# 1. 設定下載範圍與存放資料夾
start_date = datetime(2026, 10, 1)
end_date = datetime(2026, 11, 7)
raw_dir = "raw_data"
os.makedirs(raw_dir, exist_ok=True)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}

current = start_date
while current <= end_date:
    date_str = current.strftime("%Y-%m-%d")
    for hour in range(24):
        # 確保 hour 不會補零 (0, 1, ..., 23) 契合 preprocess.py
        file_name = f"{date_str}-{hour}.json.gz"
        url = f"https://data.gharchive.org/{file_name}"
        save_path = os.path.join(raw_dir, file_name)
        
        if not os.path.exists(save_path):
            try:
                print(f"Downloading {file_name}...")
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                print(f"Failed to download {file_name}: {e}")
                
    current += timedelta(days=1)

print(f"下載完成！所有檔案已存於 {raw_dir} 資料夾")