import numpy as np
import joblib
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import pandas as pd
from fracdiff import fdiff

def shape_data(df, timesteps=20, is_predict=False, is_gbm=False):
    # 移動平均、乖離度などの特徴量の計算
    df['MA_9'] = df['price_close'].rolling(window=9).mean()
    df['MA_20'] = df['price_close'].rolling(window=20).mean()
    df['MA_30'] = df['price_close'].rolling(window=30).mean()
    df['MA_50'] = df['price_close'].rolling(window=50).mean()
    df['MA_100'] = df['price_close'].rolling(window=100).mean()
    # キャンドルの特徴量の計算
    df['Upper_Wick'] = df['price_high'] - df[['price_close', 'price_open']].max(axis=1)
    df['Lower_Wick'] = df[['price_close', 'price_open']].min(axis=1) - df['price_low']
    df['Candle_Length'] = abs(df['price_close'] - df['price_open'])
    df['Green_Candle'] = (df['price_close'] > df['price_open']).astype(int)
    df = calculate_divergence_max(df)
    df['OBV'] = calculate_obv(df)
    # 新しい特徴量の追加
    df['VWAP'] = calculate_vwap(df)
    df['MFI'] = calculate_mfi(df)
    df['Volume_Oscillator'] = calculate_volume_oscillator(df)
    df['ATR'] = calc_ATR(df)
    # df['dynamic_look_back_period'] = calculate_dynamic_look_back_period(df, 'ATR', 5, 30)
    # トリプルバリアの適用
    if is_gbm:
        # df = set_labels_based_on_past_data(df, look_back_period=10, ptSl=0.01)
        # df = set_triple_barrier(df, take_profit=0.01, stop_loss=-0.01, time_horizon=10)
        # df = calc_ma_slope(df, timesteps=2, threshold=0.0001)

        # 良い感じの値: 20, 1.5, 1.5  ラベルが30%ずつに分かれる
        df = set_labels_based_on_ATR(df, look_forward_period=10, atr_multiplier_tp=1.5, atr_multiplier_sl=1.5)
    else:
        # df = set_triple_barrier(df, take_profit=0.01, stop_loss=-0.01, time_horizon=10)
        df = set_labels_based_on_ATR(df, look_forward_period=10, atr_multiplier_tp=4, atr_multiplier_sl=2)

    # 差分の計算
    columns_to_diff = ['price_close', 'MA_9', 'MA_20', 'MA_30', 'MA_50', 'MA_100', 'OBV', 'VWAP', 'divergence', 'Upper_Wick', 'Lower_Wick', 'Candle_Length']
    d = 0.5  # 例として0.5次の差分を取る
    add_fractional_diff(df, columns_to_diff, d)

    # 指定された列について異常値を検出し、置き換え
    # max divergenceは未来の値を含んでいるので注意
    columns = [
            'price_close',
            'diff_price_close',
               'diff_MA_100',
            #    'MA_100',
               'diff_MA_50',
            #    'MA_50',
               'diff_MA_30',
            #    'MA_30',
               'diff_MA_20',
            #    'MA_20',
               'diff_MA_9',
            #    'MA_9',
               'diff_divergence',
            #    'OBV',
            #    'VWAP',
               'MFI',
            #    'Volume_Oscillator',
               'ATR',
               'diff_Upper_Wick',
               'diff_Lower_Wick',
               'diff_Candle_Length',
               'Green_Candle',
            #    'volume',
            #    'turnover',
    ]
    if not is_gbm:
        for col in columns:
            replace_outliers_with_median(df, col)

    df.to_csv('./df.csv', index=False)
    if is_gbm:
        columns.append('label')
        return df[columns].copy()

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

# 分数次差分を計算し、結果を元のDataFrameに追加する関数
def add_fractional_diff(df, columns, d):
    for col in columns:
        # 分数次差分を計算
        diffed_series = fdiff(df[col].values, n=d, axis=0)
        # 新しい列名を定義
        new_col_name = f'diff_{col}'
        # 分数次差分の結果を元のDataFrameに追加
        df[new_col_name] = diffed_series
    # NaN値を持つ行を除去
    df.dropna(inplace=True)

import numpy as np

def calc_ma_slope(df, timesteps=10, threshold=0.0001):
    # ラベル列の初期化
    df['label'] = 1

    for index, row in df.iterrows():
        if index + timesteps < len(df):
            y_values = df['MA_30'].iloc[index:index+timesteps + 1].values
            x_values = np.arange(0, len(y_values))
            slope = np.polyfit(x_values, y_values, 1)[0]  # 傾きを計算

            if abs(slope) > threshold:
                if slope > 0:
                    df.at[index, 'label'] = 2  # 正の傾き
                else:
                    df.at[index, 'label'] = 0  # 負の傾き
            else:
                df.at[index, 'label'] = 1  # ほとんど傾いていない

    label_0_percentage = (df['label'] == 0).mean()
    label_1_percentage = (df['label'] == 1).mean()
    label_2_percentage = (df['label'] == 2).mean()

    print(f"label=0の割合: {label_0_percentage * 100:.2f}%")
    print(f"label=1の割合: {label_1_percentage * 100:.2f}%")
    print(f"label=2の割合: {label_2_percentage * 100:.2f}%")

    return df

def calc_ATR(df_raw, period=14):
    # 真の範囲 (TR) を計算
    df = df_raw.copy()
    df['High-Low'] = df['price_high'] - df['price_low']
    df['High-Prev Close'] = abs(df['price_high'] - df['price_close'].shift(1))
    df['Low-Prev Close'] = abs(df['price_low'] - df['price_close'].shift(1))

    df['TR'] = df[['High-Low', 'High-Prev Close', 'Low-Prev Close']].max(axis=1)

    # ATR を計算 (例えば20日間平均)
    return df['TR'].rolling(window=period).mean()


def set_triple_barrier(df, take_profit, stop_loss, time_horizon):
    # ラベル列の初期化
    df['label'] =  1

    for index, row in df.iterrows():
        # 上限バリア、下限バリア、時間バリアを設定
        upper_barrier = row['price_close'] * (1 + take_profit)
        lower_barrier = row['price_close'] * (1 + stop_loss)
        end_time = min(index + time_horizon, len(df) - 1)

        for future_index in range(index + 1, end_time + 1):
            future_row = df.iloc[future_index]
            if future_row['price_close'] >= upper_barrier:
                df.at[index, 'label'] = 2  # 上限バリア達成
                break
            elif future_row['price_close'] <= lower_barrier:
                df.at[index, 'label'] = 0  # 下限バリア達成
                break
        else:
            # 時間バリア達成
            df.at[index, 'label'] = 1

    # label=0の割合を計算
    label_0_percentage = (df['label'] == 1).mean()

    # 割合が50%以上であるかどうかを判定
    print("====================================")
    if label_0_percentage > 0.9:
        print(f"label=1の割合: {label_0_percentage * 100:.2f}%")
        print("label=1の割合が90%を超えています。")
    else:
        print(f"label=1の割合: {label_0_percentage * 100:.2f}%")
        print("label=1の割合が90%を超えていません。")

    return df

def set_labels_based_on_ATR(df, look_forward_period, atr_multiplier_tp=4, atr_multiplier_sl=2):
    df['label'] = 1  # 未定義の状態を表す初期値

    for index in range(len(df) - look_forward_period):
        base_price = df.iloc[index]['price_close']  # 基準となる現在の価格
        atr_value = df.iloc[index]['ATR']  # 現在のATR値を取得

        # 利益確定（Take Profit）と損切り（Stop Loss）の閾値を設定
        take_profit_threshold = base_price + (atr_value * atr_multiplier_tp)
        stop_loss_threshold = base_price - (atr_value * atr_multiplier_sl)

        # 未来のlook_forward_period間のデータで利益確定や損切りが発生したか確認
        for future_index in range(index + 1, index + look_forward_period + 1):
            future_price = df.iloc[future_index]['price_close']

            if future_price >= take_profit_threshold:
                df.at[index, 'label'] = 2  # 利益確定の条件を満たす
                break
            elif future_price <= stop_loss_threshold:
                df.at[index, 'label'] = 0  # 損切りの条件を満たす
                break
        else:
            df.at[index, 'label'] = 1  # その他（利益確定や損切りの条件を満たさない）

    label_0_percentage = (df['label'] == 0).mean()
    label_1_percentage = (df['label'] == 1).mean()
    label_2_percentage = (df['label'] == 2).mean()

    print(f"label=0の割合: {label_0_percentage * 100:.2f}%")
    print(f"label=1の割合: {label_1_percentage * 100:.2f}%")
    print(f"label=2の割合: {label_2_percentage * 100:.2f}%")

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
