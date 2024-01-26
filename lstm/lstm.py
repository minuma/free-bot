import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import joblib
from datetime import datetime, timedelta


def calculate_divergence_max(df):
    # 乖離度の計算
    df['divergence'] = df['MA_9'] - df['MA_100']

    # 交差点の特定
    df['cross'] = 0
    for i in range(1, len(df)):
        if (df['MA_9'].iloc[i] > df['MA_100'].iloc[i] and df['MA_9'].iloc[i - 1] <= df['MA_100'].iloc[i - 1]) or \
           (df['MA_9'].iloc[i] < df['MA_100'].iloc[i] and df['MA_9'].iloc[i - 1] >= df['MA_100'].iloc[i - 1]):
            df['cross'].iloc[i] = 1

    # 最大乖離度の計算
    max_divergence = 0
    df['max_divergence'] = 0
    for i in range(1, len(df)):
        if df['cross'].iloc[i] == 1:
            max_divergence = 0
        else:
            max_divergence = max(max_divergence, abs(df['divergence'].iloc[i]))
        df['max_divergence'].iloc[i] = max_divergence

    return df

def calculate_obv(data):
    obv = [0]
    for i in range(1, len(data)):
        if data['price_close'][i] > data['price_close'][i-1]:
            obv.append(obv[-1] + data['volume'][i])
        elif data['price_close'][i] < data['price_close'][i-1]:
            obv.append(obv[-1] - data['volume'][i])
        else:
            obv.append(obv[-1])
    obv = [x / 1000 for x in obv]
    return obv

def load_data():
    # JSONファイルからデータを読み込む
    with open('lstm/historical/csv/historical_price.json', 'r') as file:
    # with open('lstm/historical/csv/historical_price_202312.json', 'r') as file:
        data = json.load(file)

        # price_closeとvolumeをリストとして取得
        price_close_data = [item['price_close'] for item in data['data']]
        volume_data = [item['volume'] for item in data['data']]
        date = [item['date_close'] for item in data['data']]

        # Pandas DataFrameを作成
        df = pd.DataFrame({
            'price_close': price_close_data,
            'volume': volume_data,
            'date_close': date,
        })
        df['date_close'] = pd.to_datetime(df['date_close']).dt.tz_convert('Asia/Tokyo')

    # 移動平均の計算
    df['MA_9'] = df['price_close'].rolling(window=9).mean()
    df['MA_100'] = df['price_close'].rolling(window=100).mean()

    # 乖離度と最大乖離度の計算
    df = calculate_divergence_max(df)
    df['OBV'] = calculate_obv(df)

    # シフトするタイムステップの設定（例：2ステップ先を予測）
    shift_steps = 3

    # ラベル（将来のdivergence）の準備
    df['future_divergence'] = df['divergence'].shift(-shift_steps)

    # NaN値を含む行を削除
    df.dropna(inplace=True)

    # 特徴量とラベルの定義
    X = df[['price_close', 'MA_9', 'MA_100', 'divergence', 'max_divergence', 'OBV']].values
    y = df['future_divergence'].values
    df.to_csv('./df.csv', index=False)


    return X, y

def shape_data(X, y, is_predict=False):
    # データのスケーリング
    if not is_predict:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, './models/scaler.joblib')
    else:
        scaler = joblib.load('./models/scaler.joblib')
        X_scaled = scaler.transform(X)

    # 時系列データの整形
    timesteps = 24
    X_seq, y_seq = [], []
    for i in range(timesteps, len(X_scaled)):
        X_seq.append(X_scaled[i-timesteps:i])
        y_seq.append(y[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    return X_seq, y_seq

def build_model(X_seq):
    # LSTMモデルの構築
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    # 回帰問題用の出力層
    model.add(Dense(1, activation='linear'))

    # 損失関数と評価指標の変更
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mean_squared_error'])

    return model

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def validate_model(X_test, y_test):
    # モデルの読み込み
    model = load_model('./models/lstm_model.h5')

    # テストデータセットでの予測
    y_pred = model.predict(X_test)

    # y_predをNumPy配列からDataFrameに変換
    df = pd.read_csv('df.csv')
    y_pred_df = pd.DataFrame(y_pred, columns=['y_pred'])

    # df.csvの下からy_pred_dfの行数分のdate_closeを取得
    date_close = df['date_close'].iloc[-len(y_pred_df):]
    # date_closeをy_pred_dfに追加
    y_pred_df['date_close'] = date_close.reset_index(drop=True)
    # 結果をCSVに保存
    y_pred_df.to_csv('y_pred_with_date.csv', index=False)

    # 評価指標の計算
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mse)

    print(f"平均絶対誤差（MAE）: {mae}")
    print(f"平均二乗誤差（MSE）: {mse}")
    print(f"ルート平均二乗誤差（RMSE）: {rmse}")

    # オーバーフィットの確認
    # 訓練データとテストデータの評価指標を比較


if __name__ == '__main__':
    X, y = load_data()

    # データの整形
    X_seq, y_seq = shape_data(X, y, is_predict=True)

    # データのテスト
    validate_model(X_seq, y_seq)

    # # モデルの構築
    # model = build_model(X_seq)

    # # 分割の割合を定義
    # train_size = int(len(X_seq) * 0.8)
    # # 訓練データと検証データに分割
    # X_train, X_val = X_seq[:train_size], X_seq[train_size:]
    # y_train, y_val = y_seq[:train_size], y_seq[train_size:]

    # # 早期停止の設定
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # # モデルの訓練（検証セットを含む）
    # model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, callbacks=[early_stopping])

    # # モデルの保存
    # model.save('./models/lstm_model.h5')
