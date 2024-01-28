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
with open('lstm/historical/csv/historical_price_20230601.json', 'r') as file:
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
# 1次モデル（LSTM）のロード
model = load_model('./models/lstm_model.h5')
y_pred = model.predict(X_seq)

# メタモデルのロード
meta_model = load_model('./models/meta_label_model.h5')
# メタモデルの適用
selected_indices = (y_pred.flatten() > 0.5) | (y_pred.flatten() < -0.5)
X_meta = X_seq[selected_indices]
meta_predictions = meta_model.predict(X_meta)
selected_indices = np.where((y_pred.flatten() > 0.5) | (y_pred.flatten() < -0.5))[0]


# timestepsの値
timesteps = 24  # shape_data関数に合わせて調整
# dfをtimesteps分だけトリミング
df_trimmed = df.iloc[timesteps:].copy()

# y_predの長さをdf_trimmedに合わせる
length_diff_y_pred = len(df_trimmed) - len(y_pred.flatten())
adjusted_y_pred = np.concatenate([np.full(length_diff_y_pred, np.nan), y_pred.flatten()])

# meta_predictionsの全時点に対する再配置
adjusted_meta_pred = np.full(len(df_trimmed), np.nan)  # 全時点に対して初期化
adjusted_meta_pred[selected_indices] = meta_predictions.flatten()  # 選択された時点にのみ予測値を配置

# NaN値を含むadjusted_y_predを使用してsignal列を生成
adjusted_y_pred = np.nan_to_num(adjusted_y_pred, nan=0)
df_trimmed.loc[:, 'adjusted_y_pred'] = adjusted_y_pred
# TODO: 0.5以上、-0.5以下の場合に買い、売り, その他はアクションなしのシグナルを生成に変更
# df_trimmed.loc[:, 'signal'] = np.where(df_trimmed['adjusted_y_pred'] > 0.5, '買い', '売り')
df_trimmed.loc[:, 'signal'] = np.where(df_trimmed['adjusted_y_pred'] > 0.5, '買い', np.where(df_trimmed['adjusted_y_pred'] < -0.5, '売り', 'アクションなし'))



# メタモデルの予測に基づいてmeta_signal列を生成
adjusted_meta_pred = np.nan_to_num(adjusted_meta_pred, nan=0)
df_trimmed.loc[:, 'adjusted_meta_pred'] = adjusted_meta_pred
df_trimmed.loc[:, 'meta_signal'] = np.where(df_trimmed['adjusted_meta_pred'] > 0.5, df_trimmed['signal'], 'アクションなし')

# 取引シミュレーション
df_trimmed.loc[:, 'position'] = df_trimmed['meta_signal'].shift(1).replace({'買い': 1, '売り': -1, 'アクションなし': 0})
df_trimmed.loc[:, 'market_return'] = df_trimmed['price_close'].pct_change()
df_trimmed.loc[:, 'strategy_return'] = df_trimmed['market_return'] * df_trimmed['position']

# 累積リターンの計算
df_trimmed.loc[:, 'cumulative_market_return'] = (1 + df_trimmed['market_return']).cumprod()
df_trimmed.loc[:, 'cumulative_strategy_return'] = (1 + df_trimmed['strategy_return']).cumprod()
df_trimmed.to_csv('backtest_result.csv', index=False)

# 累積リターンのプロット
plt.figure(figsize=(12, 6))
plt.plot(df_trimmed['date_close'], df_trimmed['cumulative_market_return'], label='Market Return', color='red')
plt.plot(df_trimmed['date_close'], df_trimmed['cumulative_strategy_return'], label='Strategy Return', color='blue')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Backtest Result')
plt.savefig('cumulative_return_plot.png')
