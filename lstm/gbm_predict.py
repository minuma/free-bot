import pandas as pd
import lightgbm as lgb

import json
import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_data
from data_preprocessor import shape_data

def generate_trade_signal(y_pred, y_pred_meta):
    signals = []
    predicted_labels = np.argmax(y_pred, axis=1)
    # # 1つ目と3つ目の数値だけを取り出す
    # y_pred_modified = y_pred[:, [0, 2]]
    # # 修正した配列で最大値を持つインデックスを見つける
    # predicted_labels = np.argmax(y_pred_modified, axis=1)
    mean_value = np.mean(y_pred_meta)
    for pred, meta in zip(predicted_labels, y_pred_meta):
        if meta >= 0:  # y_pred_metaが0.5以上の場合のみ売買を考慮
            if pred == 2:  # メタモデルが買いを示唆する場合
                signals.append('buy')
            elif pred == 0:  # メタモデルが売りを示唆する場合
                signals.append('sell')
            else:
                signals.append('hold')
        else:
            signals.append('hold')  # y_pred_metaが0.5未満の場合は常に'hold'
    return signals


if __name__ == '__main__':
    df_predict = load_data(is_trade=True, is_bybit=True)
    df, sub_df = shape_data(df_predict, is_gbm=True)

    # モデルをファイルからロード
    df.drop(['label'], axis=1, inplace=True)
    X = df
    truncated_df = df.copy()

    # ロードしたモデルを使用して予測を実行
    loaded_model = lgb.Booster(model_file='./models/gbm/lightgbm_model.txt')
    y_pred_loaded = loaded_model.predict(X, num_iteration=loaded_model.best_iteration)


    # metaのためのデータ処理
    loaded_model_meta = lgb.Booster(model_file='./models/gbm/lightgbm_model_meta.txt')
    rows_to_drop = len(X) - len(y_pred_loaded)
    X_trimmed = X.iloc[rows_to_drop:]
    X_trimmed.reset_index(drop=True, inplace=True)

    predicted_labels = np.argmax(y_pred_loaded, axis=1)
    predictions_df = pd.DataFrame({'predicted_label': predicted_labels})
    predictions_df = predictions_df[['predicted_label']]

    X_comb = X_trimmed.join(predictions_df)
    y_pred_loaded_meta = loaded_model_meta.predict(X_comb, num_iteration=loaded_model.best_iteration, predict_disable_shape_check=True)

    # 取引シグナルに基づいて戦略リターンを計算
    truncated_df['predicted_label'] = generate_trade_signal(y_pred_loaded, y_pred_loaded_meta) 

    # df_predictのインデックスをリセット（必要に応じて）
    df_predict.reset_index(inplace=True, drop=True)

    # truncated_dfのインデックスをリセット（必要に応じて）
    truncated_df.reset_index(inplace=True, drop=True)

    # インデックスを基にして右側結合
    merged_df = pd.concat([truncated_df, df_predict[['date_close']]], axis=1)
    sub_df.reset_index(inplace=True, drop=True)
    merged_df = pd.concat([truncated_df, sub_df[['price_close', 'ATR']]], axis=1)

    # 結合後のDataFrameをCSVに保存
    merged_df.to_csv('predictions.csv', index=False)
