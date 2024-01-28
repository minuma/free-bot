import schedule
import time
import subprocess


def job():
      subprocess.run(['python', '/home/minuma/free-bot/lstm/historical/syve_api.py'], capture_output=True, text=True)
      subprocess.run(['python', '/home/minuma/free-bot/lstm/gbm_predict.py'], capture_output=True, text=True)
      subprocess.run(['python', '/home/minuma/free-bot/bot/gateio.py'], capture_output=True, text=True)

# 1分ごとにjobを実行
schedule.every(10).minutes.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
