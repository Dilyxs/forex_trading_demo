#!/bin/bash

source /root/file/venv/bin/activate

cd /root/file/forex_trading_demo


python3 func_xgb.py


echo "Python script executed successfully."

echo "[$(date)] Script executed." >> /root/file/forex_trading_demo/bot.log

