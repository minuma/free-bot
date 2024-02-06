import pandas as pd
import json
from datetime import datetime

def load_data(is_validation=False, is_backtest=False, is_trade=False):
    file_path = './lstm/historical/csv/10m/matic/historical_price_20231101.json'
    # file_path = './lstm/historical/csv/historical_price.json'
    if is_validation:
        file_path = './lstm/historical/csv/10m/matic/historical_price_20231201.json'
    if is_backtest:
        file_path = './lstm/historical/csv/10m/matic/historical_price_20240101.json'
        # file_path = './lstm/historical/csv/historical_price.json'
    if is_trade:
        file_path = './lstm/historical/csv/historical_price.json'
    # JSONファイルからデータを読み込む
    with open(file_path, 'r') as file:
        data = json.load(file)

        # price_close, volume, dateをリストとして取得
        price_close_data = [item['price_close'] for item in data['data']]
        price_open_data = [item['price_open'] for item in data['data']]
        price_high_data = [item['price_high'] for item in data['data']]
        price_low_data = [item['price_low'] for item in data['data']]
        volume_data = [item['volume'] for item in data['data']]
        date = [item['date_close'] for item in data['data']]

        # Pandas DataFrameを作成
        df = pd.DataFrame({
            'price_close': price_close_data,
            'price_open': price_open_data,
            'price_high': price_high_data,
            'price_low': price_low_data,
            'volume': volume_data,
            'date_close': date,
        })
        df['volume'] = df['volume'].apply(lambda x: abs(x) if x < 0 else x)

        # date_closeをDateTime型に変換
        df['date_close'] = pd.to_datetime(df['date_close']) + pd.Timedelta(hours=9)

    return df
