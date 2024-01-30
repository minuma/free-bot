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
from dotenv import load_dotenv
load_dotenv()

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
    contract = "MATIC_USDT"
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
    print(r.json())

def open_position(side="buy"): 
    host = "https://api.gateio.ws"
    prefix = "/api/v4"
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    url = '/futures/usdt/orders'
    query_param = ''
    contract = "MATIC_USDT"
    size = 10
    iceberg = 0
    price = "0"
    tif = "ioc"
    text = "t-my-custom-id"
    stp_act = "-"
    if side != "buy":
        size *= -1 

    body = f'{{"contract":"{contract}","size":{size},"iceberg":{iceberg},"price":"{price}","tif":"{tif}","text":"{text}","stp_act":"{stp_act}"}}'
    # for `gen_sign` implementation, refer to section `Authentication` above
    sign_headers = gen_sign('POST', prefix + url, query_param, body)
    headers.update(sign_headers)
    r = requests.request('POST', host + prefix + url, headers=headers, data=body)
    print(r.json())

def get_position_size():
    host = "https://api.gateio.ws"
    prefix = "/api/v4"
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    url = '/futures/usdt/positions/MATIC_USDT'
    query_param = ''
    # for `gen_sign` implementation, refer to section `Authentication` above
    sign_headers = gen_sign('GET', prefix + url, query_param)
    headers.update(sign_headers)
    r = requests.request('GET', host + prefix + url, headers=headers)

    return r.json().get("size")

# def get_side_from_predictions():
#     df = pd.read_csv('./predictions.csv')
#     last_predicted_value = df['predicted_label'].iloc[-1]
#     print(last_predicted_value)
#     if last_predicted_value == 0:
#         return "buy"
#     elif last_predicted_value == 2:
#         return "sell"
#     else:
#         return "hold"

def get_side_from_predictions():
    df = pd.read_csv('./predictions.csv')
    last_predicted_value = df['predicted_label'].iloc[-1]
    if last_predicted_value == 'buy':
        return "buy"
    elif last_predicted_value == 'sell':
        return "sell"
    else:
        return "hold"

if __name__ == "__main__":
    side = get_side_from_predictions()
    if side == "hold":
        print("hold")
        exit()

    now_position_size = get_position_size()
    if side == "buy" and now_position_size > 0:
        print("buy hold")
        exit()
    if side == "sell" and now_position_size < 0:
        print("sell hold")
        exit()

    if side == "buy" and now_position_size <= 0:
        close_position()
        time.sleep(1)
        print("buy open")
        open_position(side)
        exit()
    
    if side == "sell" and now_position_size >= 0:
        close_position()
        time.sleep(1)
        print("sell open")
        open_position(side)
        exit()
