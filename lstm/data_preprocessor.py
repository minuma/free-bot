import numpy as np
import joblib
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import pandas as pd



def shape_data(df, timesteps=20, is_predict=False, is_gbm=False):
    # 移動平均、乖離度などの特徴量の計算
    df['MA_9'] = df['price_close'].rolling(window=9).mean()
    df['MA_100'] = df['price_close'].rolling(window=100).mean()
    df = calculate_divergence_max(df)
    df['OBV'] = calculate_obv(df)
    # 新しい特徴量の追加
    df['VWAP'] = calculate_vwap(df)
    df['MFI'] = calculate_mfi(df)
    df['Volume_Oscillator'] = calculate_volume_oscillator(df)

    # トリプルバリアの適用
    if is_gbm:
        df = set_labels_based_on_past_data(df, look_back_period=3, ptSl=0.002)
    else:
        df = set_triple_barrier(df, take_profit=0.01, stop_loss=-0.01, time_horizon=20)

    # 差分の計算
    columns_to_diff = ['price_close', 'MA_9', 'MA_100', 'OBV', 'diff_MA_100', 'VWAP', 'divergence']
    for col in columns_to_diff:
        df[f'diff_{col}'] = df[col].diff()
    df.dropna(inplace=True)

    # 指定された列について異常値を検出し、置き換え
    columns = ['MA_100', 'price_close', 'MA_9', 'VWAP', 'divergence', 'max_divergence', 'OBV', 'MFI', 'Volume_Oscillator']
    for col in columns:
        replace_outliers_with_median(df, col)

    df.to_csv('./df.csv', index=False)
    if is_gbm:
        return df.copy()

    # 特徴量とラベルの定義
    X = df[columns].values
    # 仮定: df['label']には3つのクラスが含まれている（例えば、-1, 0, 1）
    y = pd.get_dummies(df['label']).values

    # データのスケーリング
    scaler_file = './models/scaler.joblib'
    if not is_predict:
        scaler = RobustScaler()
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

def replace_outliers_with_median(df, col):
    Q1 = df[col].quantile(0.05)
    Q3 = df[col].quantile(0.95)
    median = df[col].median()

    df[col] = np.where((df[col] < Q1) | (df[col] > Q3), median, df[col])

def set_triple_barrier(df, take_profit, stop_loss, time_horizon):
    # ラベル列の初期化
    df['label'] = 0 

    for index, row in df.iterrows():
        # 上限バリア、下限バリア、時間バリアを設定
        upper_barrier = row['price_close'] * (1 + take_profit)
        lower_barrier = row['price_close'] * (1 + stop_loss)
        end_time = min(index + time_horizon, len(df) - 1)

        for future_index in range(index + 1, end_time + 1):
            future_row = df.iloc[future_index]
            if future_row['price_close'] >= upper_barrier:
                df.at[index, 'label'] = 1  # 上限バリア達成
                break
            elif future_row['price_close'] <= lower_barrier:
                df.at[index, 'label'] = -1  # 下限バリア達成
                break
        else:
            # 時間バリア達成
            df.at[index, 'label'] = 0

    # label=0の割合を計算
    label_0_percentage = (df['label'] == 0).mean()

    # 割合が50%以上であるかどうかを判定
    print("====================================")
    if label_0_percentage > 0.9:
        print(f"label=0の割合: {label_0_percentage * 100:.2f}%")
        print("label=0の割合が90%を超えています。")
    else:
        print(f"label=0の割合: {label_0_percentage * 100:.2f}%")
        print("label=0の割合が90%を超えていません。")

    return df

def set_labels_based_on_past_data(df, look_back_period, ptSl):
    df['label'] = 0

    for index in range(look_back_period, len(df)):
        past_data = df.iloc[index - look_back_period:index]
        current_price = df.iloc[index]['price_close']

        # 過去の最高価格と最低価格を計算
        max_past_price = past_data['price_close'].max()
        min_past_price = past_data['price_close'].min()

        # 利益確定（Take Profit）と損切り（Stop Loss）の閾値を設定
        take_profit_threshold = max_past_price * (1 + ptSl)
        stop_loss_threshold = min_past_price * (1 - ptSl)

        # 現在価格が過去の最高価格に対する利益確定閾値を超えるか、
        # 過去の最低価格に対する損切り閾値を下回るかをチェック
        if current_price > take_profit_threshold:
            df.at[index, 'label'] = 2  # 利益確定の条件を満たす
        elif current_price < stop_loss_threshold:
            df.at[index, 'label'] = 0  # 損切りの条件を満たす
        else:
            df.at[index, 'label'] = 1  # その他

    return df



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
