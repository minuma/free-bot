import numpy as np
import pandas as pd
import lightgbm as lgb
from statsmodels.distributions.empirical_distribution import ECDF
from data_loader import load_data
from data_preprocessor import shape_data


## Train 
data_1 = load_data()

df_1 = shape_data(data_1, is_gbm=True)
y_1 = df_1['label']

df_1.drop(['label'], axis=1, inplace=True)
X_1 = df_1

# ##  Test
# data_test = load_data(is_validation=True)

# df_test = shape_data(data_test, is_gbm=True)
# y_test = df_test['label']

# df_test.drop(['label'], axis=1, inplace=True)
# X_test = df_test

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
    'learning_rate': 0.01,
    'num_leaves': 20,  # 少なくする
    'max_depth': 7,  # 深さを制限する
    'min_child_samples': 30,  # 増やす
    'max_bin': 255,
    'subsample': 0.6,
    'subsample_freq': 0,
    'colsample_bytree': 0.7,
    'min_child_weight': 0.001,
    'subsample_for_bin': 200000,
    'min_split_gain': 0.1,  # ゲインの最小値を設定
    'reg_alpha': 0.1,  # 正則化を少し加える
    'reg_lambda': 0.1,  # 正則化を少し加える
    'nthread': 4,
    'verbose': 0
}

# データセットの作成
train_data = lgb.Dataset(X_train, label=y_train)
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

gbm.save_model('./models/gbm/lightgbm_model.txt')

