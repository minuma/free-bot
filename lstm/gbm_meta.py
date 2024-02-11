import lightgbm as lgb
import pandas as pd
import numpy as np


# モデルをファイルからロード
loaded_model = lgb.Booster(model_file='./models/gbm/lightgbm_model_tmp.txt')

from data_loader import load_data
from data_preprocessor import shape_data


data = load_data(is_validation=True, is_bybit=True)

# 特徴量とターゲットの定義
df = shape_data(data, is_gbm=True )
y = df['label']
df.drop(['label'], axis=1, inplace=True)
X = df

# ロードしたモデルを使用して予測を実行
y_pred_loaded = loaded_model.predict(X, num_iteration=loaded_model.best_iteration)

# 各予測の最大確率を持つクラスのインデックスを取得（予測ラベル）
predicted_labels = np.argmax(y_pred_loaded, axis=1)

# 予測結果をDataFrameに変換
predictions_df = pd.DataFrame({'predicted_label': predicted_labels})

predictions_df = predictions_df[['predicted_label']]


# y（ラベル）をDataFrameに変換
labels_df = pd.DataFrame({'label': y})

labels_df_reset = labels_df.reset_index(drop=True)
predictions_df_reset = predictions_df.reset_index(drop=True)

# print(labels_df_reset.tail())
# print(predictions_df_reset.tail())

combined_df = labels_df_reset.join(predictions_df_reset)

# NaNの行を削除
combined_df.dropna(inplace=True)

# ラベルと予測値が一致するかどうかの列を追加
combined_df['is_correct'] = (combined_df['label'] == combined_df['predicted_label']).astype(int)

# 結果を確認
print(combined_df.tail())

# 予測結果をCSVファイルに保存
combined_df.to_csv('combined.csv', index=False)  # index=Falseを指定してインデックスを保存しないようにする



## train
y_comb = combined_df['is_correct']

rows_to_drop = len(X) - len(predictions_df_reset)
X_trimmed = X.iloc[rows_to_drop:]
X_trimmed.reset_index(drop=True, inplace=True)

X_comb = X_trimmed.join(predictions_df_reset)

# データをトレーニングセットとテストセットに分割
train_size = int(len(data) * 0.2)
X_train, X_test = X_comb[:train_size], X_comb[train_size:]
y_train, y_test = y_comb[:train_size], y_comb[train_size:]



params = {
    'boosting_type': 'gbdt',
    'metric': 'multi_logloss',
    'objective': 'binary',  # バイナリ分類用の目的関数
    'metric': 'binary_logloss', 
    'n_estimators': 10000, 
    'learning_rate': 0.01,
    'num_leaves': 31,  # 少なくする
    'max_depth': 10,  # 深さを制限する
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
    'verbose': -1,
    'extra_trees': True,
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

gbm.save_model('./models/gbm/lightgbm_model_meta_tmp.txt')