import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

from data_loader import load_data
from data_preprocessor import shape_data

def train():
    df = load_data()

    # データの整形
    X_seq, y_seq = shape_data(df, is_predict=False)

    # モデルの構築
    model = build_model(X_seq)

    # 分割の割合を定義
    train_size = int(len(X_seq) * 0.8)
    # 訓練データと検証データに分割
    X_train, X_val = X_seq[:train_size], X_seq[train_size:]
    y_train, y_val = y_seq[:train_size], y_seq[train_size:]

    # 早期停止の設定
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True  # 最も良いモデルの重みを復元
    )

    # モデルの訓練（検証セットを含む）
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, callbacks=[early_stopping])

    # モデルの保存
    model.save('./models/lstm_model.h5')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

def build_model(X_seq):
    model = Sequential()

    # LSTM層のサイズと数を調整
    model.add(LSTM(32, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))

    # 中間層
    model.add(Dense(16, activation='relu'))

    # 出力層にtanh活性化関数を使用
    model.add(Dense(1, activation='tanh'))

    # 損失関数をmean_absolute_errorに変更、最適化アルゴリズムをAdamに
    model.compile(loss='mean_absolute_error',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['mean_squared_error'])

    return model


from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def validate():
    df = load_data()

    # データの整形
    X_seq, y_seq = shape_data(df, is_predict=True)

    # データのテスト
    validate_model(X_seq, y_seq)

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


def filter_data_for_meta_labeling(X, y, model):
    # モデルで予測
    y_pred = model.predict(X)

    # フィルタリング条件の変更（ラベル1とラベル-1の両方を選択）
    selected_indices = np.where((y_pred > 0.5) | (y_pred < -0.5))[0]  # 閾値を調整
    if selected_indices.size == 0:
        print('No data selected for meta labeling!!!!!')
        y_pred_df = pd.DataFrame({'predicted_values': y_pred.flatten()})
        y_pred_df.to_csv('predicted_values.csv', index=False)
        return None, None

    return X[selected_indices], y[selected_indices]

def build_meta_label_model(X):
    # 新しいメタラベルモデルの構築
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 二値分類問題

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

# メイン処理
def train_meta_model():
    df = load_data()
    X_seq, y_seq = shape_data(df, is_predict=True)

    # 既存のLSTMモデルのロード
    lstm_model = load_model('./models/lstm_model.h5')

    # メタラベリングのためのデータフィルタリング
    X_meta, y_meta = filter_data_for_meta_labeling(X_seq, y_seq, lstm_model)
    if X_meta is None or y_meta is None:
        return

    # メタラベルモデルの構築
    meta_label_model = build_meta_label_model(X_meta)

    # 分割の割合を定義
    train_size = int(len(X_meta) * 0.8)
    X_train, X_val = X_meta[:train_size], X_meta[train_size:]
    y_train, y_val = y_meta[:train_size], y_meta[train_size:]

    # 早期停止の設定
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True  # 最も良いモデルの重みを復元
    )

    # メタラベルモデルの訓練
    meta_label_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, callbacks=[early_stopping])

    # モデルの保存
    meta_label_model.save('./models/meta_label_model.h5')

if __name__ == '__main__':
    train()
    # train_meta_model()
    # validate()