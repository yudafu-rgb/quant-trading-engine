import ccxt
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import ta # 技术分析库，用于生成特征

# 1. 数据库初始化 (持久化与T+7结算追踪)
def init_db():
    conn = sqlite3.connect('quant_trades.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            tx_id TEXT PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
            direction TEXT,
            entry_price REAL,
            composite_score REAL,
            settlement_time TEXT,
            status TEXT
        )
    ''')
    conn.commit()
    return conn

# 2. 实时行情拉取 (Live Price Overlay)
def fetch_live_data(symbols):
    print("⏳ 正在连接 Binance API 抓取实时现价...")
    exchange = ccxt.binance({'enableRateLimit': True})
    live_data = {}
    for symbol in symbols:
        ticker = exchange.fetch_ticker(symbol)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        live_data[symbol] = {
            'live_price': ticker['last'],
            'df': df
        }
    return live_data

# 3. 量化评分引擎 (Alpha Score Generator)
def calculate_composite_score(df):
    # 此处用 RSI 与 均线偏离度 作为基础模型特征平滑的 Proxy (替代真实的Transformer大模型)
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

    latest_rsi = df['rsi'].iloc[-1]
    latest_close = df['close'].iloc[-1]
    latest_vwap = df['vwap'].iloc[-1]

    # 模拟 HMM + SG 滤波后的复合得分归一化 [-1.0, 1.0]
    # RSI > 70 且价格高于 VWAP 倾向多头; 反之空头
    score = ((latest_rsi - 50) / 50) * 0.6 + ((latest_close - latest_vwap) / latest_vwap * 100) * 0.4
    return max(min(score, 1.0), -1.0) # 限制在 [-1, 1] 之间

# 4. 主调度函数 (Orchestrator)
def run_model():
    conn = init_db()
    cursor = conn.cursor()
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT']

    print("\n" + "="*50)
    print(f"⚙️ 量化模型运行简报 | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("="*50)

    market_data = fetch_live_data(symbols)

    longs, shorts = [], []

    for sym, data in market_data.items():
        price = data['live_price']
        score = calculate_composite_score(data['df'])

        # 强极度信号硬过滤阀门
        if score >= 0.75:
            longs.append((sym, price, score))
        elif score <= -0.75:
            shorts.append((sym, price, score))

    print("\n### 强极度做多标的 (Score >= +0.75)")
    for sym, p, s in longs:
        print(f"[+] {sym} | 现价: ${p} | 得分: {s:.2f}")
        _save_trade(cursor, conn, sym, 'LONG', p, s)

    print("\n### 强极度做空标的 (Score <= -0.75)")
    for sym, p, s in shorts:
        print(f"[-] {sym} | 现价: ${p} | 得分: {s:.2f}")
        _save_trade(cursor, conn, sym, 'SHORT', p, s)

    if not longs and not shorts:
        print("\n⚠️ 当前无标的触发强极度门限，系统全员观望 (FLAT)。")

    print("\n✅ 数据已写入 SQLite，绑定 T+7 结算。")
    conn.close()

def _save_trade(cursor, conn, symbol, direction, price, score):
    now = datetime.utcnow()
    tx_id = f"DB_{now.strftime('%Y%m%d%H%M%S')}_{symbol.replace('/','')}"
    settlement = (now + timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute(
        "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (tx_id, now.strftime('%Y-%m-%d %H:%M:%S'), symbol, direction, price, score, settlement, 'OPEN')
    )
    conn.commit()

if __name__ == "__main__":
    run_model()
