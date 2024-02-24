import numpy as np
import pandas as pd
import lightgbm as lgb
from statsmodels.distributions.empirical_distribution import ECDF
from data_loader import load_data
from data_preprocessor import shape_data
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight


## Train 
data_1 = load_data(is_bybit=True)

df_1, _ = shape_data(data_1, is_gbm=True)
y_1 = df_1['label']

df_1.drop(['label'], axis=1, inplace=True)
X_1 = df_1

# LightGBMのパラメータ設定
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'n_estimators': 10000, 
    # 'learning_rate': 0.05, # 検証用
    'learning_rate': 0.005, # 本番運用用
    'num_leaves': 30,  # 少なくする
    'max_depth': -1,  # 深さを制限する
    'min_child_samples': 20,  # 増やす
    # 'cosample_bytree': 0.7,
    # 'n_bins': 5,
    # 'max_bin': 255,
    # 'subsample': 0.6,
    # 'subsample_freq': 0,
    # 'colsample_bytree': 0.7,
    # 'min_child_weight': 0.001,
    # 'subsample_for_bin': 200000,
    # 'min_split_gain': 0.1,  # ゲインの最小値を設定
    # 'reg_alpha': 0.1,  # 正則化を少し加える
    # 'reg_lambda': 0.1,  # 正則化を少し加える
    'nthread': 5,
    'verbose': -1,
    'extra_trees': True,
    'feature_fraction': 0.05, # 低くして汎化
    'bagging_fraction': 0.05, # 低くして汎化
    'bagging_freq': 2,
    'is_unbalance': True,
}

# 時系列分割の設定
tscv = TimeSeriesSplit(n_splits=5)

# 結果を格納するリスト
accuracies = []
log_losses = []
best_iterations = []


# 分割されたデータセットでモデルを訓練・評価
for train_index, test_index in tscv.split(X_1):
    X_train, X_val = X_1.iloc[train_index], X_1.iloc[test_index]
    y_train, y_val = y_1.iloc[train_index], y_1.iloc[test_index]

    # クラスの重みを計算
    unique_classes = np.unique(y_train)  # y_trainは訓練データのラベル
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    # サンプルごとの重みを設定
    weights = np.ones(y_train.shape[0])
    for i, val in enumerate(unique_classes):
        weights[y_train == val] = class_weights[i]

    # 重み付きデータセットの作成
    # LightGBM用のデータセットを作成
    lgb_train = lgb.Dataset(X_train, y_train, weight=weights)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    # モデルの訓練
    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_val],
                    callbacks=[lgb.early_stopping(stopping_rounds=1000)])
    
    # 検証セットでの評価（例：予測と評価）
    y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)

    # 評価指標の計算（例：精度、ロスなど）
    # ここに評価コードを追加

    # 実際のラベルと予測されたラベルの最も確率が高いクラスを比較して精度を計算
    y_pred_max = np.argmax(y_pred, axis=1)  # 多クラス分類の場合、最も確率が高いクラスのインデックスを取得
    accuracy = accuracy_score(y_val, y_pred_max)
    print(f'Accuracy: {accuracy}')

    # Log Lossの計算
    logloss = log_loss(y_val, y_pred)
    print(f'Log Loss: {logloss}')

    accuracies.append(accuracy)
    log_losses.append(logloss)
    best_iterations.append(gbm.best_iteration)



# 結果のグラフ化
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Split')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(accuracies, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # 同じx軸を共有する別のy軸を作成
color = 'tab:blue'
ax2.set_ylabel('Log Loss', color=color)
ax2.plot(log_losses, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Model Performance across Splits')
fig.tight_layout()  # レイアウトの自動調整
plt.savefig('model_performance.png')


average_best_iteration = np.mean(best_iterations) * 1.2
print(f'Adjusted average best iteration: {average_best_iteration}')

# 計算された値をファイルに保存
with open('adjusted_best_iteration.txt', 'w') as f:
    f.write(str(average_best_iteration))

# 相互情報量の計算
mi = mutual_info_classif(X_train, y_train)
mi_series = pd.Series(mi, index=X_train.columns)

# 結果の表示
mi_series = mi_series.sort_values(ascending=False)

# 計算された値をファイルに保存
with open('mutual_info.txt', 'w') as f:
    f.write(str(mi_series))