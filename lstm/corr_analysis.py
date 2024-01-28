import pandas as pd

# データフレームの読み込み（あるいは作成）
df = pd.read_csv('./df.csv')

# 目的変数を含む列をデータフレームに追加
# 例: df['future_divergence'] = y

# 相関係数の計算
correlation_matrix = df.corr()

# 目的変数との相関を表示
correlation_with_target = correlation_matrix['label']
print(correlation_with_target)