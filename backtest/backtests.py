import pandas as pd
import json
import matplotlib.pyplot as plt


# データの読み込み
with open('lstm/historical/csv/historical_price.json', 'r') as file:
    data = json.load(file)

# Pandas DataFrameに変換
df = pd.DataFrame(data['data'])

# 日付列をDatetime型に変換
df['date_open'] = pd.to_datetime(df['date_open'])
df['date_close'] = pd.to_datetime(df['date_close'])

df.to_csv('back.csv', index=False)

# バックテストのロジック（ここでは単純な例を使用）
# 例: 移動平均を基にシグナルを生成
df['MA_10'] = df['price_close'].rolling(window=10).mean()
df['signal'] = df['price_close'] > df['MA_10']

# 取引シミュレーション
df['position'] = df['signal'].diff()
df['daily_return'] = df['price_close'].pct_change()
df['strategy_return'] = df['daily_return'] * df['position'].shift(1)

# 累積リターンの計算
cumulative_return = (df['strategy_return'].cumsum() + 1)

# 累積リターンのプロット
cumulative_return.plot()
plt.savefig('cumulative_return_plot.png')
