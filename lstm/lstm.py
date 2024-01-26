import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def load_data():
    # JSONファイルからデータを読み込む
    with open('lstm/historical/csv/historical_price.json', 'r') as file:
        data = json.load(file)
        price_data = [item['price_close'] for item in data['data']]

    # Pandas DataFrameを作成
    df = pd.DataFrame(price_data, columns=['price_close'])

    # 移動平均の計算
    df['MA_9'] = df['price_close'].rolling(window=9).mean()
    df['MA_100'] = df['price_close'].rolling(window=100).mean()

    # NaN値を取り除く
    df.dropna(inplace=True)

    # 特徴量とラベルの準備
    X = df[['price_close', 'MA_9', 'MA_100']].values
    y = df['price_close'].values

    return X, y

def shape_data(X, y):
    # データのスケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 時系列データの整形
    timesteps = 20 
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
    model.add(LSTM(50, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    return model

X, y = load_data()
X_seq, y_seq = shape_data(X, y)
model = build_model(X_seq)

# モデルのコンパイル
model.compile(optimizer='adam', loss='mean_squared_error')

# モデルの訓練
model.fit(X_seq, y_seq, epochs=10, batch_size=32)

# モデルの保存
model.save('lstm_model.h5')
