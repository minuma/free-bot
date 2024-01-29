import schedule
import time
import subprocess

def job():
    with open('/home/minuma/free-bot/result.txt', 'a') as f:  # ここで指定したパスに結果を追記します
        # 各コマンドの実行結果をファイルに書き込む
        result = subprocess.run(['python', '/home/minuma/free-bot/lstm/historical/syve_api.py'], capture_output=True, text=True)
        f.write(result.stdout)

        result = subprocess.run(['python', '/home/minuma/free-bot/lstm/gbm_predict.py'], capture_output=True, text=True)
        # f.write(result.stdout + "\n")

        result = subprocess.run(['python', '/home/minuma/free-bot/bot/gateio.py'], capture_output=True, text=True)
        f.write(result.stdout)

# スケジュール設定前に1回実行
job()

# 10分ごとにjobを実行
schedule.every(10).minutes.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
