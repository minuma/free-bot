# example authentication implementation in Python

"""
Python SDK is recommended as it has already implemented the authentication process for every API:
"""
import pandas as pd
import time
import hashlib
import hmac
import requests
import json
import os
import math
from dotenv import load_dotenv
load_dotenv()

contract = "MATIC_USDT"
multiplier = 1.5 # ATRの何倍でTP, SPを設定するか

def gen_sign(method, url, query_string=None, payload_string=None):
    key = os.getenv('GATE_IO_API_KEY')        # api_key
    secret = os.getenv('GATE_IO_API_SECRET')      # api_secret

    t = time.time()
    m = hashlib.sha512()
    m.update((payload_string or "").encode('utf-8'))
    hashed_payload = m.hexdigest()
    s = '%s\n%s\n%s\n%s\n%s' % (method, url, query_string or "", hashed_payload, t)
    sign = hmac.new(secret.encode('utf-8'), s.encode('utf-8'), hashlib.sha512).hexdigest()
    return {'KEY': key, 'Timestamp': str(t), 'SIGN': sign}

def close_position():
    host = "https://api.gateio.ws"
    prefix = "/api/v4"
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    url = '/futures/usdt/orders'
    query_param = ''
    size = 0
    iceberg = 0
    price = "0"
    tif = "ioc"
    text = "t-my-custom-id"
    stp_act = "-"

    body = f'{{"contract":"{contract}","size":{size},"iceberg":{iceberg},"price":"{price}","tif":"{tif}","text":"{text}","stp_act":"{stp_act}","close":true}}'
    # for `gen_sign` implementation, refer to section `Authentication` above
    sign_headers = gen_sign('POST', prefix + url, query_param, body)
    headers.update(sign_headers)
    r = requests.request('POST', host + prefix + url, headers=headers, data=body)

def open_position(side="buy", size=1): 
    host = "https://api.gateio.ws"
    prefix = "/api/v4"
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    url = '/futures/usdt/orders'
    query_param = ''
    iceberg = 0
    price = "0"
    tif = "ioc"
    text = "t-my-custom-id"
    stp_act = "-"
    size
    if side != "buy":
        size *= -1 

    body = f'{{"contract":"{contract}","size":{size},"iceberg":{iceberg},"price":"{price}","tif":"{tif}","text":"{text}","stp_act":"{stp_act}"}}'
    # for `gen_sign` implementation, refer to section `Authentication` above
    sign_headers = gen_sign('POST', prefix + url, query_param, body)
    headers.update(sign_headers)
    r = requests.request('POST', host + prefix + url, headers=headers, data=body)

import requests
import json

def price_tp_sp_orders(side, now_price, ATR):
    now_price_float = float(now_price)
    if side == "buy":
        tp_trigger_price = now_price_float + ATR * multiplier
        formatted_string = format(tp_trigger_price, '.4f')
        price_trigger_order(formatted_string, rule=1, side=side)

        time.sleep(1)

        sp_trigger_price = now_price_float - ATR * multiplier
        formatted_string = format(sp_trigger_price, '.4f')
        price_trigger_order(formatted_string, rule=2, side=side)
    else:
        tp_trigger_price = now_price_float - ATR * multiplier
        formatted_string = format(tp_trigger_price, '.4f')
        price_trigger_order(formatted_string, rule=2, side=side)

        time.sleep(1)

        sp_trigger_price = now_price_float + ATR * multiplier
        formatted_string = format(sp_trigger_price, '.4f')
        price_trigger_order(formatted_string, rule=1, side=side)
    

def price_trigger_order(trigger_price, rule=1, side="buy"):
    # https://www.gate.io/docs/developers/apiv4/#create-a-price-triggered-order-2
    host = "https://api.gateio.ws"
    prefix = "/api/v4"
    url = '/futures/usdt/price_orders'
    query_param = ''
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    auto_size = 'close_long'
    if side != "buy":
        auto_size = 'close_short'
    # 注文のパラメータを設定
    body = json.dumps({
        "initial": {
            "contract": contract,
            "size": 0,
            "tif": "ioc",
            "close": False,
            "price": '0',
            "reduce_only": True,
            "auto_size": auto_size,
        },
        "trigger": {
            "strategy_type": 0,
            "price_type": 0,
            "price": trigger_price,
            "rule": rule, # tp, slで方向を変える
            "expiration": 0
        },
    })

    # 認証ヘッダーを生成（gen_sign関数の実装が必要）
    sign_headers = gen_sign('POST', prefix + url, query_param, body)
    headers.update(sign_headers)

    # APIリクエストを実行
    r = requests.post(host + prefix + url, headers=headers, data=body)

def get_mark_price():
    host = "https://api.gateio.ws"
    prefix = "/api/v4"
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    url = '/futures/usdt/contracts/' + contract
    query_param = ''
    r = requests.request('GET', host + prefix + url, headers=headers)
    return r.json().get("mark_price")

def get_position_size():
    host = "https://api.gateio.ws"
    prefix = "/api/v4"
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    url = '/futures/usdt/positions/' + contract
    query_param = ''
    # for `gen_sign` implementation, refer to section `Authentication` above
    sign_headers = gen_sign('GET', prefix + url, query_param)
    headers.update(sign_headers)
    r = requests.request('GET', host + prefix + url, headers=headers)

    return r.json().get("size")

def list_price_orders():
    host = "https://api.gateio.ws"
    prefix = "/api/v4"
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    url = '/futures/usdt/price_orders'
    query_param = 'status=open'
    # for `gen_sign` implementation, refer to section `Authentication` above
    sign_headers = gen_sign('GET', prefix + url, query_param)
    headers.update(sign_headers)
    r = requests.request('GET', host + prefix + url + "?" + query_param, headers=headers)
    print(r.json())

def cancel_price_orders():
    host = "https://api.gateio.ws"
    prefix = "/api/v4"
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    url = '/futures/usdt/price_orders'
    query_param = 'contract=' + contract
    # for `gen_sign` implementation, refer to section `Authentication` above
    sign_headers = gen_sign('DELETE', prefix + url, query_param)
    headers.update(sign_headers)
    r = requests.request('DELETE', host + prefix + url + "?" + query_param, headers=headers)

def get_side_from_predictions():
    df = pd.read_csv('./predictions.csv')
    last_predicted_value = df['predicted_label'].iloc[-1]
    diff_price_close = df['diff_MA_9'].iloc[-1]
    ATR = df['ATR'].iloc[-1]
    # diff_VWAP = df['diff_VWAP'].iloc[-2]
    if diff_price_close == 0:
        return 'close', ATR
    else:
        return last_predicted_value, ATR

def calculate_position_size(usd_amount_allocated_per_atr, trade_unit, atr, usd_price):
    position_size = usd_amount_allocated_per_atr / (trade_unit * (atr * multiplier) * float(usd_price))
    return math.floor(position_size)

if __name__ == "__main__":
    side, ATR = get_side_from_predictions()
    now_mark_price = get_mark_price()
    print(side)
    size = calculate_position_size(0.5, 10, ATR, now_mark_price)
    if side == "hold":
        print("hold")
        close_position()
        cancel_price_orders()
        exit()

    if side == "close":
        print("close")
        close_position()
        cancel_price_orders()
        exit()

    now_position_size = get_position_size()
    if side == "buy" and now_position_size > 0:
        print("buy hold")
        # cancel_price_orders()
        # time.sleep(1)
        # price_tp_sp_orders(side, now_mark_price, ATR)
        exit()
    if side == "sell" and now_position_size < 0:
        print("sell hold")
        # cancel_price_orders()
        # time.sleep(1)
        # price_tp_sp_orders(side, now_mark_price, ATR)
        exit()

    if side == "buy" and now_position_size <= 0:
        close_position()
        time.sleep(1)
        cancel_price_orders()
        time.sleep(1)
        print("buy open")
        open_position(side, size)
        time.sleep(1)
        # price_tp_sp_orders(side, now_mark_price, ATR)
        exit()
    
    if side == "sell" and now_position_size >= 0:
        close_position()
        time.sleep(1)
        cancel_price_orders()
        time.sleep(1)
        print("sell open")
        open_position(side, size)
        time.sleep(1)
        # price_tp_sp_orders(side, now_mark_price, ATR)
        exit()
