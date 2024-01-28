import pandas as pd
import json
from datetime import datetime

def load_data():
    # JSONファイルからデータを読み込む
    with open('lstm/historical/csv/2h/historical_price_20230101.json', 'r') as file:
        data = json.load(file)

        # price_close, volume, dateをリストとして取得
        price_close_data = [item['price_close'] for item in data['data']]
        volume_data = [item['volume'] for item in data['data']]
        date = [item['date_close'] for item in data['data']]

        # Pandas DataFrameを作成
        df = pd.DataFrame({
            'price_close': price_close_data,
            'volume': volume_data,
            'date_close': date,
        })

        # date_closeをDateTime型に変換
        df['date_close'] = pd.to_datetime(df['date_close']).dt.tz_convert('Asia/Tokyo')

    return df
