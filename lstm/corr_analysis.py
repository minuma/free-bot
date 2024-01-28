import pandas as pd
from sklearn.model_selection import train_test_split


# データフレームの読み込み（あるいは作成）
df = pd.read_csv('./df.csv')
df.drop(columns=['date_close'], inplace=True)

# 特徴量とラベルの分離
X = df.drop('label', axis=1)
y = df['label']

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# ランダムフォレストモデルのインスタンス作成
model = RandomForestClassifier(n_estimators=100, random_state=42)

# モデルのトレーニング
model.fit(X_train, y_train)

import matplotlib.pyplot as plt
import seaborn as sns

# 特徴量の重要性を取得
importances = model.feature_importances_

# 重要性の高い順に特徴量名を取得
indices = importances.argsort()[::-1]
features = X.columns[indices]

# 重要性の高いトップ5の特徴量を表示
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices][:10], y=features[:10])
plt.title('Top 10 Important Features')
plt.show()
plt.savefig('corr.png')
