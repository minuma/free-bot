import numpy as np
import joblib
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import pandas as pd
from fracdiff import fdiff
import talib

def shape_data(df, timesteps=20, is_predict=False, is_gbm=False):
    close = df['price_close'].values
    open = df['price_open'].values
    high = df['price_high'].values
    low = df['price_low'].values
    volume = df['volume'].astype(float).values

    df['MA_5'] = df['price_close'].rolling(window=5).mean()
    df['MA_9'] = talib.SMA(close, timeperiod=9)
    df['MA_20'] = df['price_close'].rolling(window=20).mean()
    df['MA_30'] = df['price_close'].rolling(window=30).mean()
    df['MA_50'] = df['price_close'].rolling(window=50).mean()
    df['MA_75'] = df['price_close'].rolling(window=75).mean()
    df['MA_100'] = df['price_close'].rolling(window=100).mean()
    # キャンドルの特徴量の計算
    df['Upper_Wick'] = df['price_high'] - df[['price_close', 'price_open']].max(axis=1)
    df['Lower_Wick'] = df[['price_close', 'price_open']].min(axis=1) - df['price_low']
    df['Candle_Length'] = abs(df['price_close'] - df['price_open'])
    df['Green_Candle'] = (df['price_close'] > df['price_open']).astype(int)
    # df = calculate_divergence_max(df)
    # 新しい特徴量の追加
    df['VWAP'] = calculate_vwap(df)
    df['Volume_Oscillator'] = calculate_volume_oscillator(df)
    df['ATR'] = calc_ATR(df)

    # ランダムなノイズデータを追加
    np.random.seed(42)  # 再現性のための乱数シード
    df['noise'] = np.random.normal(0, 1, len(df))  # 平均0、標準偏差1の正規分布からノイズを生成
    
    df, overlap_added_columns = calc_overlap_studies(df, close, open, high, low)
    df, momentum_added_columns = calc_momentum_indicators(df, open, high, low, close, volume)
    df, volume_added_columns = calc_volume_indicators(df, high, low, close, volume)
    df, volatility_added_columns = calc_volatility_indicators(df, high, low, close)
    df, cycle_added_columns = calc_cycle_indicators(df, close)
    df, statistic_added_columns = calc_statistic_functions(df, high, low, close)
    # df, pattern_columns = calc_pattern_recognition(df, open, high, low, close)

    # トリプルバリアの適用
    if is_gbm:
        # 良い感じの値: 20, 1.5, 1.5  ラベルが30%ずつに分かれる
        # df = set_labels_based_on_ATR(df, look_forward_period=20, atr_multiplier_tp=4, atr_multiplier_sl=4)
        df = set_labels_based_on_MA_slope(df, n=5, slope_threshold=1)
        # df = set_labels_based_on_ATR_added(df, look_forward_period=20, atr_multiplier_tp=3, atr_multiplier_sl=3, look_forward_extension_period=5)
        # df = classify_future_direction(df, look_forward_period=15, threshold=0.002)
    else:
        # df = set_triple_barrier(df, take_profit=0.01, stop_loss=-0.01, time_horizon=10)
        df = set_labels_based_on_ATR(df, look_forward_period=10, atr_multiplier_tp=4, atr_multiplier_sl=2)

    # 差分の計算
    # columns_to_diff = ['price_close', 'MA_5', 'MA_9', 'MA_20', 'MA_30', 'MA_50', 'MA_75', 'MA_100', 'VWAP',  'MFI', 'ATR', 'Volume_Oscillator', 'noise']
    columns_to_diff = ['Upper_Wick', 'Lower_Wick', 'Candle_Length', 'price_close', 'MA_5', 'MA_9', 'MA_20', 'MA_30', 'MA_50', 'MA_75', 'MA_100', 'VWAP',  'MFI', 'ATR', 'Volume_Oscillator', 'noise']
    d = 0.8  # 少ないほど過去の情報を多く含む
    add_fractional_diff(df, columns_to_diff, d)

    # 指定された列について異常値を検出し、置き換え
    # max divergenceは未来の値を含んでいるので注意
    columns = [
            # 'diff_price_close',
               'diff_MA_100',
               'diff_MA_75',
               'diff_MA_50',
               'diff_MA_30',
               'diff_MA_20',
               'diff_MA_9',
               'diff_MA_5',
               'diff_VWAP', # yとの相関が高すぎる
            #    'diff_MFI',
               'diff_Volume_Oscillator',
               'diff_ATR',
               'diff_Upper_Wick',
               'diff_Lower_Wick',
            #    'diff_Candle_Length',
            #    'Green_Candle',
            #    'volume',
            #    'turnover',
                # 'diff_noise',
    ]
    columns_sub = [
            'price_close',
            'ATR',
    ]
    added_columns = overlap_added_columns + momentum_added_columns + volume_added_columns + volatility_added_columns + statistic_added_columns + cycle_added_columns
    # added_columns =  momentum_added_columns

    # added_columns = ['NATR', 'ATR', 'ADXR', 'TRIX', 'PLUS_DM', 'OBV', 'CORREL', 'AD', 'CMO', 'DX', 'macdsignalext']
    columns += added_columns
    add_fractional_diff_raw(df, added_columns, d)

    if not is_gbm:
        for col in columns:
            replace_outliers_with_median(df, col)

    df.to_csv('./df.csv', index=False)
    if is_gbm:
        columns.append('label')
        return df[columns].copy(), df[columns_sub].copy()

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

# 分数次差分を計算し、結果を元のDataFrameに追加する関数
def add_fractional_diff_raw(df, columns, d):
    for col in columns:
        # 分数次差分を計算
        diffed_series = fdiff(df[col].values, n=d, axis=0)
        # 新しい列名を定義
        # 分数次差分の結果を元のDataFrameに追加
        df[col] = diffed_series
    # NaN値を持つ行を除去
    df.dropna(inplace=True)

import numpy as np

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

def set_labels_based_on_ATR_added(df, look_forward_period, atr_multiplier_tp=4, atr_multiplier_sl=2, look_forward_extension_period=5):
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
                # 利益確定後の追加の価格動向を確認
                for extension_index in range(future_index + 1, min(future_index + 1 + look_forward_extension_period, len(df))):
                    if df.iloc[extension_index]['price_close'] > future_price:
                        df.at[index, 'label'] = 4  # 利益確定後にさらに上昇
                        break
                break
            elif future_price <= stop_loss_threshold:
                df.at[index, 'label'] = 0  # 損切りの条件を満たす
                # 損切り後の追加の価格動向を確認
                for extension_index in range(future_index + 1, min(future_index + 1 + look_forward_extension_period, len(df))):
                    if df.iloc[extension_index]['price_close'] < future_price:
                        df.at[index, 'label'] = 3  # 損切り後にさらに下落
                        break
                break

    label_0_percentage = (df['label'] == 0).mean()
    label_1_percentage = (df['label'] == 1).mean()
    label_2_percentage = (df['label'] == 2).mean()
    label_3_percentage = (df['label'] == 3).mean()
    label_4_percentage = (df['label'] == 4).mean()

    print(f"label=0の割合: {label_0_percentage * 100:.2f}%")
    print(f"label=1の割合: {label_1_percentage * 100:.2f}%")
    print(f"label=2の割合: {label_2_percentage * 100:.2f}%")
    print(f"label=3の割合: {label_3_percentage * 100:.2f}%")
    print(f"label=4の割合: {label_4_percentage * 100:.2f}%")

    return df

import numpy as np

def set_labels_based_on_MA_slope(df, n=25, slope_threshold=0.1):
    # 未来のn期間での移動平均の傾きを計算するために、まず未来の価格平均を計算
    df['future_MA'] = df['price_close'].rolling(window=n).mean().shift(-n)

    # 未来の移動平均の傾きを計算
    df['future_MA_slope'] = (df['future_MA'] - df['price_close']) / n

    # 傾きに基づいてラベルを設定
    df['label'] = np.where(df['future_MA_slope'] > slope_threshold, 2,  # 上昇トレンド
                           np.where(df['future_MA_slope'] < -slope_threshold, 0,  # 下降トレンド
                                    1))  # 変化なし

    # 最後のn行は未来のデータがないためラベルを設定できない
    df['label'].iloc[-n:] = np.nan  # 最後のn期間は未来のデータがないのでNaNを割り当てる

    # 各ラベルの割合を計算
    label_0_percentage = (df['label'] == 0).mean()
    label_1_percentage = (df['label'] == 1).mean()
    label_2_percentage = (df['label'] == 2).mean()

    print(f"label=0の割合: {label_0_percentage * 100:.2f}%")
    print(f"label=1の割合: {label_1_percentage * 100:.2f}%")
    print(f"label=2の割合: {label_2_percentage * 100:.2f}%")

    return df


def classify_future_direction(df, column='price_close', look_forward_period=5, threshold=0.01):
    """
    未来の価格方向を分類します。
    df: データフレーム
    column: 価格データのカラム名
    look_forward_period: 未来を予測する期間
    threshold: 分類のための閾値
    """
    future_return = df[column].shift(-look_forward_period) / df[column] - 1
    df['label'] = 1  # 価格変動なしを示すデフォルト値
    df.loc[future_return > threshold, 'label'] = 2  # 価格上昇
    df.loc[future_return < -threshold, 'label'] = 0  # 価格下落

    label_0_percentage = (df['label'] == 0).mean()
    label_1_percentage = (df['label'] == 1).mean()
    label_2_percentage = (df['label'] == 2).mean()

    print(f"label=0の割合: {label_0_percentage * 100:.2f}%")
    print(f"label=1の割合: {label_1_percentage * 100:.2f}%")
    print(f"label=2の割合: {label_2_percentage * 100:.2f}%")
    
    return df


def calculate_vwap(data):
    vwap = (data['price_close'] * data['volume']).cumsum() / data['volume'].cumsum()
    return vwap

def calculate_volume_oscillator(data, short_period=5, long_period=10):
    short_ma = data['volume'].rolling(window=short_period).mean()
    long_ma = data['volume'].rolling(window=long_period).mean()
    vo = (short_ma - long_ma) / long_ma
    return vo

def calc_overlap_studies(df, close, open, high, low):
    added_columns = [
        'upperband', 'middleband', 'lowerband', 
        'DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'MA', 
        'mama', 'fama', 'MIDPOINT', 'MIDPRICE', 
        'SAR', 'SAREXT', 'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA'
    ]
    exclude_columns = ['SAR', 'SAREXT', 'T3', 'upperband', 'middleband', 'lowerband']
    added_columns = [col for col in added_columns if col not in exclude_columns]

    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['upperband'], df['middleband'], df['lowerband'] = upperband, middleband, lowerband

    df['DEMA'] = talib.DEMA(close, timeperiod=30)
    df['EMA'] = talib.EMA(close, timeperiod=30)
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close)
    df['KAMA'] = talib.KAMA(close, timeperiod=30)
    df['MA'] = talib.MA(close, timeperiod=30, matype=0)
    mama, fama = talib.MAMA(close, fastlimit=0.5, slowlimit=0.05) # fastlimitとslowlimitに仮の値を使用
    df['mama'], df['fama'] = mama, fama
    # df['MAVP'] = talib.MAVP(close, periods, minperiod=2, maxperiod=30, matype=0) # 'periods'は適切な期間の配列を指す変数
    df['MIDPOINT'] = talib.MIDPOINT(close, timeperiod=14)
    df['MIDPRICE'] = talib.MIDPRICE(high, low, timeperiod=14)
    df['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2) # accelerationとmaximumにデフォルト値を設定
    df['SAREXT'] = talib.SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2)
    df['SMA'] = talib.SMA(close, timeperiod=30)
    df['T3'] = talib.T3(close, timeperiod=5, vfactor=0.7) # vfactorに仮の値を使用
    df['TEMA'] = talib.TEMA(close, timeperiod=30)
    df['TRIMA'] = talib.TRIMA(close, timeperiod=30)
    df['WMA'] = talib.WMA(close, timeperiod=30)
    return df, added_columns

def calc_momentum_indicators(df, open, high, low, close, volume):
    added_columns = [
        'ADX', 'ADXR', 'APO', 'aroondown', 'aroonup', 'AROONOSC', 'BOP', 'CCI', 'CMO', 'DX',
        'macd', 'macdsignal', 'macdhist', 'macdext', 'macdsignalext', 'macdhistext', 'macdfix', 'macdsignalfix', 'macdhistfix',
        'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI',
        'slowk', 'slowd', 'fastk', 'fastd', 'stochrsik', 'stochrsid', 'TRIX', 'ULTOSC', 'WILLR'
    ]
    exclude_columns = ['RSI', 'CMO', 'stochrsik', 'CCI', 'CCI', 'PLUS_DM', 'aroondown', 'macd', 'WILLR', 'aroonup', 'AROONOSC', 'BOP', 'DX', 'macdsignal', 'slowk', 'MFI', 'ULTOSC', 'fastd', 'slowd', 'stochrsid', 'fastk']
    added_columns = [col for col in added_columns if col not in exclude_columns]

    df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
    df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    aroondown, aroonup = talib.AROON(high, low, timeperiod=14)
    df['aroondown'], df['aroonup'] = aroondown, aroonup
    df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
    df['BOP'] = talib.BOP(open, high, low, close)
    df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
    df['CMO'] = talib.CMO(close, timeperiod=14)
    df['DX'] = talib.DX(high, low, close, timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'], df['macdsignal'], df['macdhist'] = macd, macdsignal, macdhist
    macd, macdsignal, macdhist = talib.MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    df['macdext'], df['macdsignalext'], df['macdhistext'] = macd, macdsignal, macdhist
    macd, macdsignal, macdhist = talib.MACDFIX(close, signalperiod=9)
    df['macdfix'], df['macdsignalfix'], df['macdhistfix'] = macd, macdsignal, macdhist
    df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
    df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)
    df['MOM'] = talib.MOM(close, timeperiod=10)
    df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)
    df['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    df['ROC'] = talib.ROC(close, timeperiod=10)
    df['ROCP'] = talib.ROCP(close, timeperiod=10)
    df['ROCR'] = talib.ROCR(close, timeperiod=10)
    df['ROCR100'] = talib.ROCR100(close, timeperiod=10)
    df['RSI'] = talib.RSI(close, timeperiod=14)
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['slowk'], df['slowd'] = slowk, slowd
    fastk, fastd = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['fastk'], df['fastd'] = fastk, fastd
    fastk, fastd = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['stochrsik'], df['stochrsid'] = fastk, fastd
    df['TRIX'] = talib.TRIX(close, timeperiod=30)
    df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

    return df, added_columns

def calc_volume_indicators(df, high, low, close, volume):
    added_columns = [
        'AD', 'ADOSC', 'OBV'
    ]
    exclude_columns = ['ADOSC']
    added_columns = [col for col in added_columns if col not in exclude_columns]

    df['AD'] = talib.AD(high, low, close, volume)
    df['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    df['OBV'] = talib.OBV(close, volume)
    
    return df, added_columns


def calc_volatility_indicators(df, high, low, close):
    added_columns = [
        'ATR', 'NATR', 'TRANGE'
    ]
    exclude_columns = ['TRANGE', 'NATR']
    added_columns = [col for col in added_columns if col not in exclude_columns]

    df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    df['NATR'] = talib.NATR(high, low, close, timeperiod=14)
    df['TRANGE'] = talib.TRANGE(high, low, close)
    
    return df, added_columns

def calc_cycle_indicators(df, close):
    added_columns = [
        'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR_inphase', 'HT_PHASOR_quadrature', 'HT_SINE_sine', 'HT_SINE_leadsine', 'HT_TRENDMODE'
    ]
    exclude_columns = ['HT_PHASOR_inphase', 'HT_DCPHASE', 'HT_TRENDMODE', 'HT_DCPERIOD', 'HT_SINE_leadsine']
    added_columns = [col for col in added_columns if col not in exclude_columns]

    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
    df['HT_DCPHASE'] = talib.HT_DCPHASE(close)
    inphase, quadrature = talib.HT_PHASOR(close)
    df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = inphase, quadrature
    sine, leadsine = talib.HT_SINE(close)
    df['HT_SINE_sine'], df['HT_SINE_leadsine'] = sine, leadsine
    df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)
    
    return df, added_columns

def calc_statistic_functions(df, high, low, close):
    added_columns = [
        'BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR'
    ]
    exclude_columns = ['BETA', 'CORREL', 'LINEARREG_INTERCEPT', 'STDDEV', 'LINEARREG_ANGLE', 'LINEARREG_SLOPE', 'VAR']
    added_columns = [col for col in added_columns if col not in exclude_columns]

    df['BETA'] = talib.BETA(high, low, timeperiod=5)
    df['CORREL'] = talib.CORREL(high, low, timeperiod=30)
    df['LINEARREG'] = talib.LINEARREG(close, timeperiod=14)
    df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close, timeperiod=14)
    df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(close, timeperiod=14)
    df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close, timeperiod=14)
    df['STDDEV'] = talib.STDDEV(close, timeperiod=5, nbdev=1)
    df['TSF'] = talib.TSF(close, timeperiod=14)
    df['VAR'] = talib.VAR(close, timeperiod=5, nbdev=1)
    
    return df, added_columns

def calc_pattern_recognition(df, open, high, low, close):
    # パターン認識関数のリスト
    pattern_functions = [
        'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
        'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
        'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
        'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
        'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
        'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
        'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE',
        'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK',
        'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM',
        'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW',
        'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK',
        'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES',
        'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN',
        'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR',
        'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
    ]

    # 各パターン認識関数を実行し、結果をDataFrameに追加
    for func_name in pattern_functions:
        # TA-Lib関数の取得と実行
        func = getattr(talib, func_name)
        df[func_name] = func(open, high, low, close)    

    return df, pattern_functions