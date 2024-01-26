import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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

    # 乖離度と最大乖離度の計算
    df = calculate_divergence_max(df)

    # シフトするタイムステップの設定（例：2ステップ先を予測）
    shift_steps = 2

    # ラベル（将来のdivergence）の準備
    df['future_divergence'] = df['divergence'].shift(-shift_steps)

    # NaN値を含む行を削除
    df.dropna(inplace=True)

    # 特徴量とラベルの定義
    X = df[['price_close', 'MA_9', 'MA_100', 'divergence', 'max_divergence']].values
    y = df['future_divergence'].values
    
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
