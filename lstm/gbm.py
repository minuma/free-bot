import numpy as np
import pandas as pd
import lightgbm as lgb
from statsmodels.distributions.empirical_distribution import ECDF
from data_loader import load_data
from data_preprocessor import shape_data
import matplotlib.pyplot as plt



## Train 
data_1 = load_data(is_bybit=True)

df_1, _ = shape_data(data_1, is_gbm=True)
y_1 = df_1['label']

df_1.drop(['label'], axis=1, inplace=True)
X_1 = df_1


# ##  Test
# data_test = load_data(is_validation=True, is_bybit=True)

# df_test, _ = shape_data(data_test, is_gbm=True)
# y_test = df_test['label']

# df_test.drop(['label'], axis=1, inplace=True)
# X_test = df_test
# X_test.to_csv('./X_test.csv', index=False)

# データをトレーニングセットとテストセットに分割
train_size = int(len(data_1) * 0.2)
X_train, X_test = X_1[:train_size], X_1[train_size:]
y_train, y_test = y_1[:train_size], y_1[train_size:]

# LightGBMのパラメータ設定
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'n_estimators': 50000, 
    'learning_rate': 0.05,
    'num_leaves': 31,  # 少なくする
    'max_depth': 30,  # 深さを制限する
    'min_child_samples': 50,  # 増やす
    'max_bin': 255,
    'subsample': 0.6,
    'subsample_freq': 0,
    'colsample_bytree': 0.7,
    'min_child_weight': 0.001,
    'subsample_for_bin': 200000,
    'min_split_gain': 0.1,  # ゲインの最小値を設定
    'reg_alpha': 1.0,  # 正則化を少し加える
    'reg_lambda': 1.0,  # 正則化を少し加える
    'nthread': 4,
    'verbose': -1,
    'extra_trees': True,
    'feature_fraction': 0.09, # 低くして汎化
    'bagging_fraction': 0.09, # 低くして汎化 n_estimatorsを使い切らないくらいに設定
    'bagging_freq': 20,
}

# データセットの作成
train_data = lgb.Dataset(X_1, label=y_1)
# train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# モデルのトレーニング
gbm = lgb.train(
    params, 
    train_data, 
    valid_sets=[train_data, test_data], 
    num_boost_round=5000, 
    callbacks=[
        lgb.early_stopping(stopping_rounds=500),
    ]
)

# テストデータに対する予測
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

gbm.save_model('./models/gbm/lightgbm_model_tmp.txt')

# 特徴量の重要度をプロット
ax = lgb.plot_importance(gbm, max_num_features=10, importance_type='gain')
plt.title("Feature Importance")
plt.xlabel("Feature Importance")
plt.ylabel("Features")

# プロットを画像ファイルとして保存
plt.savefig('feature_importance.png')
