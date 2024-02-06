import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np


from tensorflow.keras.models import load_model

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from lstm.data_loader import load_data
from lstm.data_preprocessor import shape_data

def generate_trade_signal(y_pred, meta_predictions):
    signals = []
    for pred, meta_pred in zip(y_pred, meta_predictions):
        if meta_pred > 0.5:  # メタモデルが取引を示唆する場合
            if pred[0] > pred[2]:  # 1次モデルが「買い」を示唆する場合
                signals.append('buy')
            elif pred[2] > pred[0]:  # 1次モデルが「売り」を示唆する場合
                signals.append('sell')
            else:
                signals.append('hold')
        else:
            signals.append('hold')  # メタモデルが取引を示唆しない場合
    return signals

def calculate_strategy_return(signals, market_returns):
    strategy_returns = [0]  # 初期値は0、最初のエントリでリターンはない
    position = 0  # 初期ポジション

    for i in range(1, len(signals)):
        if signals.iloc[i-1] == 'buy':
            position = 1
        elif signals.iloc[i-1] == 'sell':
            position = -1
        elif signals.iloc[i-1] == 'hold':
            # 'hold'の場合、前回のポジションを維持
            position = 0

        # 戦略リターンは1個遅れた市場リターンとポジションに依存する
        strategy_return = market_returns.iloc[i] * position
        strategy_returns.append(strategy_return)

    return strategy_returns


if __name__ == '__main__':
    # データの読み込み
    with open('lstm/historical/csv/10m/historical_price_20240101.json', 'r') as file:
        data = json.load(file)

    # Pandas DataFrameに変換
    df = pd.DataFrame(data['data'])

    # 日付列をDatetime型に変換
    df['date_open'] = pd.to_datetime(df['date_open'])
    df['date_close'] = pd.to_datetime(df['date_close'])
    df['market_return'] = df['price_close'].pct_change()
    df['market_return'].fillna(0, inplace=True)

    # バックテストのロジック（ここでは単純な例を使用）
    # 例: 移動平均を基にシグナルを生成
    df_predict = load_data()
    X_seq, y_seq = shape_data(df_predict, is_predict=True)
    # 1次モデル（LSTM）のロード
    model = load_model('./models/lstm_model.h5')
    y_pred = model.predict(X_seq)

    # メタモデルのロード
    meta_model = load_model('./models/meta_label_model.h5')
    # メタモデルの適用
    meta_predictions = meta_model.predict(X_seq)

    # 予測の結合（具体的なロジックは要調整）
    truncated_df = df.iloc[:len(y_pred)]
    truncated_df['meta_predictions'] = meta_predictions
    truncated_df['trade_signal'] = generate_trade_signal(y_pred, meta_predictions) 

    # 取引シグナルに基づいて戦略リターンを計算
    truncated_df['strategy_return'] = calculate_strategy_return(truncated_df['trade_signal'], truncated_df['market_return']) 

    # 累積リターンの計算
    truncated_df['cumulative_market_return'] = (1 + truncated_df['market_return']).cumprod()
    truncated_df['cumulative_strategy_return'] = (1 + truncated_df['strategy_return']).cumprod()

    # 結果をCSVファイルに保存
    truncated_df.to_csv('backtest_result.csv', index=False)

    # 累積リターンのプロット
    plt.figure(figsize=(12, 6))
    plt.plot(truncated_df['date_close'], truncated_df['cumulative_market_return'], label='Market Return', color='red')
    plt.plot(truncated_df['date_close'], truncated_df['cumulative_strategy_return'], label='Strategy Return', color='blue')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Backtest Result')
    plt.savefig('cumulative_return_plot.png')
