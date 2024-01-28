import numpy as np
import pandas as pd
import lightgbm as lgb
from statsmodels.distributions.empirical_distribution import ECDF
from data_loader import load_data
from data_preprocessor import shape_data


data = load_data()

# 特徴量とターゲットの定義
# 例: 'close' をターゲットとする場合
df = shape_data(data, is_df=True)
y = df['label']
X = df.drop(['label'], axis=1) # 'close' 以外の列を特徴量とする
X = df.drop(['date_close'], axis=1) # 'close' 以外の列を特徴量とする

# データをトレーニングセットとテストセットに分割
train_size = int(len(data) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LightGBMのパラメータ設定
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'max_bin': 255,
    'subsample': 0.6,
    'subsample_freq': 0,
    'colsample_bytree': 0.7,
    'min_child_weight': 0.001,
    'subsample_for_bin': 200000,
    'min_split_gain': 0,
    'reg_alpha': 0,
    'reg_lambda': 0,
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
        lgb.early_stopping(stopping_rounds=50),
    ]
)

# テストデータに対する予測
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

gbm.save_model('./models/bgm/lightgbm_model.txt')

