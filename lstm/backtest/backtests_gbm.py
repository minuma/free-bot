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
    print(y_pred)
    # # 1つ目と3つ目の数値だけを取り出す
    # y_pred_modified = y_pred[:, [0, 2]]
    # # 修正した配列で最大値を持つインデックスを見つける
    # predicted_labels = np.argmax(y_pred_modified, axis=1)
    mean_value = np.mean(y_pred_meta)
    for pred, meta in zip(predicted_labels, y_pred_meta):
        if meta >= 0:  # y_pred_metaが0.5以上の場合のみ売買を考慮
            if pred == 2:  # メタモデルが買いを示唆する場合
                signals.append('buy')
            elif pred == 0:  # メタモデルが売りを示唆する場合
                signals.append('sell')
            else:
                signals.append('hold')
        else:
            signals.append('hold')  # y_pred_metaが0.5未満の場合は常に'hold'
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
    # バックテストのロジック（ここでは単純な例を使用）
    # 例: 移動平均を基にシグナルを生成
    df_predict = load_data(is_backtest=True, is_bybit=True)
    df = shape_data(df_predict, is_gbm=True)

    # モデルをファイルからロード
    df.drop(['label'], axis=1, inplace=True)

    truncated_df = df.copy()
    truncated_df['date_close'] = df_predict['date_close']

    X = df

    # ロードしたモデルを使用して予測を実行
    loaded_model = lgb.Booster(model_file='./models/gbm/lightgbm_model_tmp.txt')
    y_pred_loaded = loaded_model.predict(X, num_iteration=loaded_model.best_iteration)


    # metaのためのデータ処理
    loaded_model_meta = lgb.Booster(model_file='./models/gbm/lightgbm_model_meta_tmp.txt')
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

   # 買いシグナルと売りシグナルのプロット
    # buy_dates = truncated_df.loc[truncated_df['trade_signal'] == 'buy', 'date_close']
    # sell_dates = truncated_df.loc[truncated_df['trade_signal'] == 'sell', 'date_close']
    # plt.scatter(buy_dates, truncated_df.loc[truncated_df['trade_signal'] == 'buy', 'cumulative_strategy_return'], label='Buy Signal', marker='^', color='green')
    # plt.scatter(sell_dates, truncated_df.loc[truncated_df['trade_signal'] == 'sell', 'cumulative_strategy_return'], label='Sell Signal', marker='v', color='black')

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Backtest Result')
    plt.savefig('cumulative_return_plot.png')

    # 'trade_signal'列の値の割合を計算
    trade_signal_counts = truncated_df['trade_signal'].value_counts(normalize=True)

    # 割合をパーセンテージとして出力
    print("Buy: {:.2%}".format(trade_signal_counts.get('buy', 0)))
    print("Sell: {:.2%}".format(trade_signal_counts.get('sell', 0)))
    print("Hold: {:.2%}".format(trade_signal_counts.get('hold', 0)))
