from datetime import datetime, timedelta
import time
import requests
import pandas as pd

# https://bybit-exchange.github.io/docs/v5/market/kline
url = "https://api.bybit.com/v5/market/kline"
symbol = "MATICUSDT"
category = "linear"
interval =3 

# 特定の開始日を設定（例：2023年1月1日）
start_date_str = "2023-04-01"
start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

# 1ヶ月後の日付を計算
end_date = start_date + timedelta(days=180)

# UNIXタイムスタンプに変換
start_timestamp = int(start_date.timestamp()) * 1000
end_timestamp = int(end_date.timestamp()) * 1000

values = []

# 終了タイムスタンプを超えないようにする
current_start_timestamp = start_timestamp
while current_start_timestamp < end_timestamp:
    params = {
        "symbol": symbol,
        "interval": interval,
        "category": category,
        "start": current_start_timestamp,
        "end": min((current_start_timestamp + 200 * 60 * interval * 1000), end_timestamp),
        "limit": 200
    }

    response = requests.get(url, params=params)
    response_data = response.json()

    if len(response_data["result"]["list"]) == 0:
        break

    original_list = response_data["result"]["list"]
    reversed_list = original_list[::-1]
    values += reversed_list
    # 最後のデータのタイムスタンプを新しい開始点として設定
    last_timestamp = int(values[-1][0]) 
    if (last_timestamp == current_start_timestamp):
        break
    # print(values[-1])
    current_start_timestamp = last_timestamp
    print(current_start_timestamp)

data = pd.DataFrame(values)

# カラム名を修正
data.columns = [
    "timestamp",
    "price_open",
    "price_high",
    "price_low",
    "price_close",
    "volume",
    "turnover"
]
# タイムスタンプをdatetime型に変換
data["date_close"] = (data['timestamp'].astype(int)/1000).apply(datetime.fromtimestamp)
data['date_close'] = pd.to_datetime(data['date_close']) + pd.Timedelta(hours=9)
# 昇順に並び替え
data.sort_values("date_close", inplace=True)

# # "datetime"をインデックスに設定
# data.set_index("datetime", inplace=True)

new_date_str = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%Y%m%d")
# data.to_csv(f"./lstm/historical/bybit_{symbol}_{category}_{interval}_{start_date_str}_to_{end_date.strftime('%Y-%m-%d')}.csv", index=False)
data.to_csv(f"./lstm/historical/historical_price_{new_date_str}.csv", index=False)
