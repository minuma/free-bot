import requests
import json
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta

load_dotenv()

class SyveApi:
    def __init__(self, token_address, pool_address):
        # https://syve.readme.io/reference/price_historical_ohlc
        self.base_url = 'https://api.syve.ai/v1/price/historical/ohlc'
        self.token_address = token_address
        self.pool_address = pool_address

    def get_historical_price(self):
        # date_str = "2023-02-01:00:00Z"
        # # datetimeオブジェクトに変換
        # date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        # # UTCタイムゾーンを設定
        # date_obj = date_obj.replace(tzinfo=timezone.utc)
        # # Unixタイムスタンプに変換
        # from_timestamp = int(date_obj.timestamp())

        # ↓ 実取引用
        # 現在の日時を取得
        now = datetime.now(timezone.utc)
        # 1日前の日時を計算
        three_days_ago = now - timedelta(days=30)
        # Unixタイムスタンプに変換
        from_timestamp = int(three_days_ago.timestamp())

        params = {
            'token_address': self.token_address,
            'pool_address': self.pool_address,
            'interval': '2h',
            'max_size': 2500,
            'fill': True,
            'key': os.getenv('SYVE_API_KEY'),
            'from_timestamp': from_timestamp,
            # 'until_timestamp': until_timestamp,
            'order': 'asc', # 昇順
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # ステータスコードが200番台でない場合に例外を発生させる
            return response.json()  # JSONデータを返す
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTPエラーが発生しました: {http_err}")
        except Exception as err:
            print(f"予期せぬエラーが発生しました: {err}")
        return None

# クラスの使用例
token_address = '0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0' # arb
pool_address = 'all'

api = SyveApi(token_address, pool_address)
historical_price = api.get_historical_price()

if historical_price:
    # フォルダの存在を確認し、存在しない場合は作成
    csv_folder_path = './lstm/historical/csv'
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)

    # JSONファイルに保存
    try:
        with open(os.path.join(csv_folder_path, 'historical_price.json'), 'w') as json_file:
            json.dump(historical_price, json_file, indent=4)
        print("データがcsv/historical_price.jsonに保存されました。")
    except Exception as e:
        print(f"ファイル保存中にエラーが発生しました: {e}")
else:
    print("データの取得に失敗しました。")