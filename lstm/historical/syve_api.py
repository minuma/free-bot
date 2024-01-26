import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

class SyveApi:
    def __init__(self, token_address, pool_address):
        self.base_url = 'https://api.syve.ai/v1/price/historical/ohlc'
        self.token_address = token_address
        self.pool_address = pool_address

    def get_historical_price(self):
        params = {
            'token_address': self.token_address,
            'pool_address': self.pool_address,
            'interval': '10m',
            'max_size': 1000,
            'key': os.getenv('API_KEY')
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
token_address = '0xf819d9cb1c2a819fd991781a822de3ca8607c3c9' # unibot
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
            json.dump(historical_price, json_file)
        print("データがcsv/historical_price.jsonに保存されました。")
    except Exception as e:
        print(f"ファイル保存中にエラーが発生しました: {e}")
else:
    print("データの取得に失敗しました。")