import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

from data_loader import load_data
from data_preprocessor import shape_data

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


if __name__ == '__main__':
    df = load_data()

    # データの整形
    X_seq, y_seq = shape_data(df, is_predict=True)

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
