import schedule
import time
import subprocess
from datetime import datetime


def job():
    with open('/home/minuma/free-bot/result.txt', 'a') as f:  # ここで指定したパスに結果を追記します
        # 各コマンドの実行結果をファイルに書き込む
        result = subprocess.run(['python', '/home/minuma/free-bot/lstm/historical/syve_api.py'], capture_output=True, text=True)
        f.write(result.stdout)

        result = subprocess.run(['python', '/home/minuma/free-bot/lstm/gbm_predict.py'], capture_output=True, text=True)
        f.write(result.stdout + "\n")

        result = subprocess.run(['python', '/home/minuma/free-bot/bot/gateio.py'], capture_output=True, text=True)
        f.write(result.stdout)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"current time: {current_time}\n\n")

# スケジュール設定前に1回実行
# job()

# 毎時特定の分に実行するスケジュールを設定
# for minute in range(1, 60, 10):
    # schedule.every().hour.at(f":{minute:02d}").do(job)

minute = 1
schedule.every().hour.at(f":{minute:02d}").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
