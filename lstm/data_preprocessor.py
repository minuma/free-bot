import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def shape_data(df, timesteps=24, is_predict=False):
    # 移動平均、乖離度などの特徴量の計算
    df['MA_9'] = df['price_close'].rolling(window=9).mean()
    df['MA_100'] = df['price_close'].rolling(window=100).mean()
    df = calculate_divergence_max(df)
    df['OBV'] = calculate_obv(df)
    # 新しい特徴量の追加
    df['VWAP'] = calculate_vwap(df)
    df['MFI'] = calculate_mfi(df)
    df['Volume_Oscillator'] = calculate_volume_oscillator(df)
    
    # シフトするタイムステップの設定（例：3ステップ先を予測）
    shift_steps = 3
    df['future_divergence'] = df['divergence'].shift(-shift_steps)
    df.dropna(inplace=True)

    # 特徴量とラベルの定義
    # TODO: OBVは値の変化量が大きすぎるため、学習には不適
    X = df[['price_close', 'MA_9', 'MA_100', 'divergence', 'max_divergence', 'VWAP', 'MFI', 'Volume_Oscillator']].values
    y = df['future_divergence'].values
    df.to_csv('./df.csv', index=False)


    # データのスケーリング
    scaler_file = './models/scaler.joblib'
    if not is_predict:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_file)
    else:
        scaler = joblib.load(scaler_file)
        X_scaled = scaler.transform(X)

    # 時系列データの整形
    X_seq, y_seq = [], []
    for i in range(timesteps, len(X_scaled)):
        X_seq.append(X_scaled[i-timesteps:i])
        y_seq.append(y[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq) 

    return X_seq, y_seq


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
    obv = [x / 1000000 for x in obv]
    return obv


def calculate_vwap(data):
    vwap = (data['price_close'] * data['volume']).cumsum() / data['volume'].cumsum()
    return vwap

def calculate_mfi(data, period=14):
    # 1日変化量
    delta = data['price_close'].diff()

    # 値上がりと値下がり
    up = delta.clip(lower=0) * data['volume']
    down = -1 * delta.clip(upper=0) * data['volume']

    # 値上がりと値下がりの合計
    up_sum = up.rolling(window=period).sum()
    down_sum = down.rolling(window=period).sum()

    # MFIの計算
    mfi = 100 - (100 / (1 + (up_sum / down_sum)))
    return mfi

def calculate_volume_oscillator(data, short_period=5, long_period=10):
    short_ma = data['volume'].rolling(window=short_period).mean()
    long_ma = data['volume'].rolling(window=long_period).mean()
    vo = (short_ma - long_ma) / long_ma
    return vo
