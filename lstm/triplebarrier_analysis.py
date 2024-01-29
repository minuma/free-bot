from data_loader import load_data

def set_triple_barrier(df, take_profit, stop_loss, time_horizon):
    # ラベル列の初期化
    df['label'] = 0 

    label_values = []  # ラベルを格納するリストを初期化

    for index, row in df.iterrows():
        # 上限バリア、下限バリア、時間バリアを設定
        upper_barrier = row['price_close'] * (1 + take_profit)
        lower_barrier = row['price_close'] * (1 + stop_loss)
        end_time = min(index + time_horizon, len(df) - 1)

        for future_index in range(index + 1, end_time + 1):
            future_row = df.iloc[future_index]
            if future_row['price_close'] >= upper_barrier:
                label_values.append(1)  # 上限バリア達成
                break
            elif future_row['price_close'] <= lower_barrier:
                label_values.append(-1)  # 下限バリア達成
                break
        else:
            # 時間バリア達成
            label_values.append(0)

    # ラベル列を追加
    df['label'] = label_values

    # label=0の割合を計算
    label_0_percentage = (df['label'] == 0).mean()

    return label_0_percentage

def set_labels_based_on_past_data(df, look_back_period, ptSl):
    df['label'] = 1  # 初期値を設定（-1は未定義を意味する）

    label_values = []  # ラベルを格納するリストを初期化
    label_values = [-1] * look_back_period
    for index in range(look_back_period, len(df)):
        # look_back_period前の価格を基準価格とする
        base_price = df.iloc[index - look_back_period]['price_close']

        # 利益確定（Take Profit）と損切り（Stop Loss）の閾値を設定
        take_profit_threshold = base_price * (1 + ptSl)
        stop_loss_threshold = base_price * (1 - ptSl)

        # 現在から過去look_back_period間のデータで利益確定や損切りが発生したか確認
        for past_index in range(index - look_back_period + 1, index + 1):
            past_price = df.iloc[past_index]['price_close']

            if past_price >= take_profit_threshold:
                label_values.append(2)
                break
            elif past_price <= stop_loss_threshold:
                label_values.append(0)
                break
        else:
            label_values.append(1)

    # label=0の割合を計算
    df['label'] = label_values
    label_1_percentage = (df['label'] == 1).mean()

    return label_1_percentage

import pandas as pd

def calculate_percentage(df, take_profit, stop_loss, time_horizon):
    # set_triple_barrier関数を呼び出してpercentageを計算
    percentage = set_triple_barrier(df, take_profit, stop_loss, time_horizon)

    # 計算されたpercentageを追加
    return percentage

def calculate_percentage_2(df, ptsl, time_horizon):
    # set_triple_barrier関数を呼び出してpercentageを計算
    percentage = set_labels_based_on_past_data(df, time_horizon, ptsl)

    # 計算されたpercentageを追加
    return percentage

if __name__ == '__main__':
    # 結果を保存するテーブル
    result_table = pd.DataFrame(columns=['Take Profit', 'Stop Loss', 'Time Horizon', 'Percentage'])

    # 異なる引数の組み合わせに対して計算
    take_profit_values = [0.005, 0.01, 0.02, 0.03]
    time_horizon_values = [5, 10, 15, 20]

    df = load_data()
    for take_profit in take_profit_values:
        for time_horizon in time_horizon_values:
            # percentage = calculate_percentage(df, take_profit, -take_profit, time_horizon)
            percentage = calculate_percentage_2(df, take_profit, time_horizon)

            # 結果をテーブルに追加
            result_table = pd.concat([result_table, pd.DataFrame({'Take Profit': [take_profit],
                                                                  'Stop Loss': [-take_profit],
                                                                  'Time Horizon': [time_horizon],
                                                                  'Percentage': [percentage]})])

    # 結果をCSVファイルとして保存
    result_table.to_csv('percentage_results.csv', index=False)
