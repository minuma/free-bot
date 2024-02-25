import numpy as np
import pandas as pd
import lightgbm as lgb
from statsmodels.distributions.empirical_distribution import ECDF
from data_loader import load_data
from data_preprocessor import shape_data
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight


## Train 
data_1 = load_data(is_bybit=True)

df_1, _ = shape_data(data_1, is_gbm=True)
y_train = df_1['label']

df_1.drop(['label'], axis=1, inplace=True)
X_train = df_1


# #  Test
# data_test = load_data(is_validation=True, is_bybit=True)

# df_test, _ = shape_data(data_test, is_gbm=True)
# y_test = df_test['label']

# df_test.drop(['label'], axis=1, inplace=True)
# X_test = df_test
# X_test.to_csv('./X_test.csv', index=False)

# # # データをトレーニングセットとテストセットに分割
# train_size = int(len(data_1) * 0.8)
# X_train, X_test = X_1[:train_size], X_1[train_size:]
# y_train, y_test = y_1[:train_size], y_1[train_size:]

# LightGBMのパラメータ設定
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'n_estimators': 600, 
    # 'learning_rate': 0.05, # 検証用
    'learning_rate': 0.01, # 本番運用用
    'num_leaves': 5,  # 少なくする
    'max_depth': 5,  # 深さを制限する
    'min_child_samples': 100,  # 増やす
    # 'cosample_bytree': 0.5,
    # 'n_bins': 5,
    # 'max_bin': 255,
    # 'subsample': 0.4,
    # 'subsample_freq': 0,
    # 'colsample_bytree': 0.7,
    # 'min_child_weight': 0.001,
    # 'subsample_for_bin': 200000,
    # 'min_split_gain': 0.1,  # ゲインの最小値を設定
    'reg_alpha': 3.5,  # 正則化を少し加える
    'reg_lambda': 3.5,  # 正則化を少し加える
    'nthread': 5,
    'verbose': -1,
    'extra_trees': True,
    'feature_fraction': 0.5, # 低くして汎化
    'bagging_fraction': 0.5, # 低くして汎化
    'bagging_freq': 8,
    'is_unbalance': True,
}


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

# モデルのトレーニング
gbm = lgb.train(
    params, 
    lgb_train, 
    # valid_sets=[train_data, test_data], 
    # num_boost_round=5000, 
    # callbacks=[
    #     lgb.early_stopping(stopping_rounds=100),
    # ]
)

# テストデータに対する予測
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

gbm.save_model('./models/gbm/lightgbm_model_tmp.txt')

# 特徴量の重要度をプロット
ax = lgb.plot_importance(gbm, max_num_features=20, importance_type='split')
plt.title("Feature Importance")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()  # レイアウトの自動調整


# プロットを画像ファイルとして保存
plt.savefig('feature_importance.png')
