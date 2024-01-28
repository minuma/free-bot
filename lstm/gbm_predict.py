import lightgbm as lgb
import pandas as pd


# モデルをファイルからロード
loaded_model = lgb.Booster(model_file='./models/gbm/lightgbm_model.txt')

from data_loader import load_data
from data_preprocessor import shape_data


data = load_data(is_validation=True)

# 特徴量とターゲットの定義
# 例: 'close' をターゲットとする場合
df = shape_data(data, is_df=True)
y = df['label']
date = df['date_close']
X = df.drop(['label'], axis=1) # 'close' 以外の列を特徴量とする
X = df.drop(['date_close'], axis=1) # 'close' 以外の列を特徴量とする

# ロードしたモデルを使用して予測を実行
y_pred_loaded = loaded_model.predict(X, num_iteration=loaded_model.best_iteration)

# 予測結果をDataFrameに変換
predictions_df = pd.DataFrame({'predicted_value': y_pred_loaded})

# 'date' 列を 'predictions_df' に追加
predictions_df['date'] = date.values

# 'date' 列を特定の位置に移動（ここでは最初の列に移動）
predictions_df = predictions_df[['date', 'predicted_value']]

# 予測結果をCSVファイルに保存
predictions_df.to_csv('predictions.csv', index=False)  # index=Falseを指定してインデックスを保存しないようにする