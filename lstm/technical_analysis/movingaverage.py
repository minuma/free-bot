import numpy as np

class MovingAverage:
    def __init__(self, period=10):
        self.period = period
        self.data = []

    def calc_moving_average(self, data):
        # データの初期化
        self.data = data
        ma = [0] * (self.period - 1)

        for idx in range(self.period, len(data) + 1):
            window = data[idx - self.period:idx]
            ma.append(np.average(window))

        return np.array(ma)

    def update_moving_average(self, value):
        # 新しいデータポイントを追加し、最古のデータポイントを削除
        if len(self.data) >= self.period:
            self.data.pop(0)
        self.data.append(value)

        # 更新されたデータで移動平均を計算
        if len(self.data) == self.period:
            return np.average(self.data)
        else:
            return None
