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


# データの読み込み
with open('lstm/historical/csv/historical_price.json', 'r') as file:
    data = json.load(file)

# Pandas DataFrameに変換
df = pd.DataFrame(data['data'])

# 日付列をDatetime型に変換
df['date_open'] = pd.to_datetime(df['date_open'])
df['date_close'] = pd.to_datetime(df['date_close'])

# バックテストのロジック（ここでは単純な例を使用）
# 例: 移動平均を基にシグナルを生成
df_predict = load_data()
X_seq, y_seq = shape_data(df_predict, is_predict=True)
model = load_model('./models/lstm_model.h5')
y_pred = model.predict(X_seq)

# timestepsの値
timesteps = 24  # shape_data関数に合わせて調整
# dfをtimesteps分だけトリミング
df_trimmed = df.iloc[timesteps:]

# 予測値をDataFrameに変換
y_pred_df = pd.DataFrame(y_pred, columns=['predicted_value'])
# df_trimmedとy_pred_dfの長さの差を計算
length_diff = len(df_trimmed) - len(y_pred_df)

# y_pred_dfの先頭にNaN行を追加して長さを合わせる
nan_rows = pd.DataFrame({'predicted_value': [np.nan] * length_diff})
y_pred_df_extended = pd.concat([nan_rows, y_pred_df], ignore_index=True)

# df_trimmedに予測値を追加
df_trimmed = df_trimmed.reset_index(drop=True)
df_trimmed['predicted_value'] = y_pred_df_extended
# df_trimmed['diff'] = df_trimmed['predicted_value'].diff()

# def generate_signal(row):
#     if row['diff'] > 0:
#         if row['predicted_value'] < 0 and abs(row['predicted_value']) < some_threshold:
#             return '売り'
#         else:
#             return '買い'
#     elif row['diff'] < 0:
#         # if row['predicted_value'] > 0 and abs(row['predicted_value']) < some_threshold:
#         #     return '買い'
#         # else:
#         return '売り'
#     else:
#         return 'アクションなし'

# # some_thresholdには、買いシグナルを生成するための閾値を設定
# some_threshold = 0.008

# トレンドを識別するための移動平均
df_trimmed['predicted_ma'] = df_trimmed['predicted_value'].rolling(window=5).mean()

def generate_combined_signal(row):
    # トレンドの識別
    if row['predicted_ma'] > upper_trend_threshold:
        trend = '上向き'
    elif row['predicted_ma'] < lower_trend_threshold:
        trend = '下向き'
    else:
        trend = '無し'

    # 逆張り条件
    if trend == '上向き' and row['predicted_value'] < row['predicted_ma'] * (1 - reversal_threshold):
        return '買い'
    elif trend == '下向き' and row['predicted_value'] > row['predicted_ma'] * (1 + reversal_threshold):
        return '売り'

    # ブレイクアウト条件
    if row['predicted_value'] > breakout_threshold:
        return '買い'
    elif row['predicted_value'] < breakdown_threshold:
        return '売り'

    return 'アクションなし'

# 閾値の設定
upper_trend_threshold = 0.01
lower_trend_threshold = 0.01
reversal_threshold = 0.05  # 5%の逆張り閾値
breakout_threshold = 0.01
breakdown_threshold = 0.01

# シグナルの生成
# df['signal'] = df.apply(generate_combined_signal, axis=1)


df_trimmed['signal'] = df_trimmed.apply(generate_combined_signal, axis=1)

# # バックテストのロジック
# df_trimmed['signal'] = df_trimmed['predicted_value'].diff().apply(
#     lambda x: '買い' if x > 0 else ('売り' if x < 0 else 'アクションなし')
# )

df_trimmed.to_csv('backtest.csv')

# 取引シミュレーション
df_trimmed['position'] = df_trimmed['signal'].shift(1)
df_trimmed['market_return'] = df_trimmed['price_close'].pct_change()
df_trimmed['strategy_return'] = df_trimmed['market_return'] * df_trimmed['position'].apply(
    lambda x: 1 if x == '買い' else (-1 if x == '売り' else 0)
)

# 累積リターンの計算
df_trimmed['cumulative_market_return'] = (1 + df_trimmed['market_return']).cumprod()
df_trimmed['cumulative_strategy_return'] = (1 + df_trimmed['strategy_return']).cumprod()

# 累積リターンのプロット
plt.figure(figsize=(12, 6))
plt.plot(df_trimmed['cumulative_market_return'], label='market return', color='red')
plt.plot(df_trimmed['cumulative_strategy_return'], label='strategy return', color='blue')
plt.legend()
plt.savefig('cumulative_return_plot.png')
