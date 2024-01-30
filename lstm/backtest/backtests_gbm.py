import pandas as pd
import lightgbm as lgb

import json
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from lstm.data_loader import load_data
from lstm.data_preprocessor import shape_data

def generate_trade_signal(y_pred, y_pred_meta):
    signals = []
    predicted_labels = np.argmax(y_pred, axis=1)
    mean_value = np.mean(y_pred_meta)
    for pred, meta in zip(predicted_labels, y_pred_meta):
        if meta >= mean_value:  # y_pred_metaが0.5以上の場合のみ売買を考慮
            if pred == 0:  # メタモデルが買いを示唆する場合
                signals.append('buy')
            elif pred == 2:  # メタモデルが売りを示唆する場合
                signals.append('sell')
            else:
                signals.append('hold')
        else:
            signals.append('hold')  # y_pred_metaが0.5未満の場合は常に'hold'
    return signals


def calculate_strategy_return(signals, market_returns):
    strategy_returns = []
    position = 0  # 現在のポジション（買い=1、売り=-1、ホールド=0）

    for signal, market_return in zip(signals, market_returns):
        if signal == 'buy':
            position = 1
        elif signal == 'sell':
            position = -1
        else:
            position = 0

        # 戦略リターンは市場リターンとポジションに依存する
        strategy_return = market_return * position
        strategy_returns.append(strategy_return)

    return strategy_returns


if __name__ == '__main__':
    # データの読み込み
    with open('lstm/historical/csv/10m/historical_price_20230301.json', 'r') as file:
        data = json.load(file)

    # Pandas DataFrameに変換
    df_raw = pd.DataFrame(data['data'])

    # 日付列をDatetime型に変換
    df_raw['date_open'] = pd.to_datetime(df_raw['date_open'])
    df_raw['date_close'] = pd.to_datetime(df_raw['date_close'])

    # バックテストのロジック（ここでは単純な例を使用）
    # 例: 移動平均を基にシグナルを生成
    df_predict = load_data(is_backtest=True)
    df = shape_data(df_predict, is_gbm=True)

    # モデルをファイルからロード
    y = df['label']
    truncated_df = df.copy()
    truncated_df['date_open'] = df_raw['date_open']
    truncated_df['date_close'] = df_raw['date_close']

    df.drop(['label'], axis=1, inplace=True)
    X = df

    # ロードしたモデルを使用して予測を実行
    loaded_model = lgb.Booster(model_file='./models/gbm/lightgbm_model.txt')
    y_pred_loaded = loaded_model.predict(X, num_iteration=loaded_model.best_iteration)


    # metaのためのデータ処理
    loaded_model_meta = lgb.Booster(model_file='./models/gbm/lightgbm_model_meta.txt')
    rows_to_drop = len(X) - len(y_pred_loaded)
    X_trimmed = X.iloc[rows_to_drop:]
    X_trimmed.reset_index(drop=True, inplace=True)

    predicted_labels = np.argmax(y_pred_loaded, axis=1)
    predictions_df = pd.DataFrame({'predicted_label': predicted_labels})
    predictions_df = predictions_df[['predicted_label']]

    X_comb = X_trimmed.join(predictions_df)
    y_pred_loaded_meta = loaded_model_meta.predict(X_comb, num_iteration=loaded_model.best_iteration)

    # 取引シグナルに基づいて戦略リターンを計算
    truncated_df['trade_signal'] = generate_trade_signal(y_pred_loaded, y_pred_loaded_meta) 

    truncated_df['market_return'] = truncated_df['price_close'].pct_change()
    truncated_df['market_return'].fillna(0, inplace=True)
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
