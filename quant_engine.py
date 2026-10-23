import ccxt
import pandas as pd
import ta
import os
import tempfile
from datetime import datetime, timedelta, timezone

CSV_FILE = 'quant_trades.csv'
HOLDING_PERIOD_HOURS = 24
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT']
HEADER = "tx_id,timestamp,symbol,direction,entry_price,rsi,dev_atr,vwap_z,score,settlement_time,exit_price,pnl_pct,status\n"


def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w') as f:
            f.write(HEADER)


def fetch_and_prep_data(exchange, symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.iloc[:-1].copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['ema_24'] = ta.trend.EMAIndicator(df['close'], window=24).ema_indicator()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    typical = (df['high'] + df['low'] + df['close']) / 3
    rolling_vol = df['volume'].rolling(window=24).sum()
    df['rolling_vwap'] = (df['volume'] * typical).rolling(window=24).sum() / rolling_vol
    df['close_std'] = df['close'].rolling(window=24).std()
    return df


def calculate_confluence_score(df):
    latest = df.iloc[-1]
    rsi = latest['rsi']
    ema = latest['ema_24']
    atr = latest['atr']
    if pd.isna(rsi) or pd.isna(ema) or pd.isna(atr) or atr == 0:
        vwap_z = 0.0
        if 'rolling_vwap' in df.columns and 'close_std' in df.columns:
            rv = latest['rolling_vwap']
            sd = latest['close_std']
            if not pd.isna(rv) and not pd.isna(sd) and sd != 0:
                vwap_z = round(float((latest['close'] - rv) / sd), 2)
        return 0.0, rsi, 0.0, vwap_z
    dev_atr = (latest['close'] - ema) / atr
    rv = latest['rolling_vwap']
    sd = latest['close_std']
    if pd.isna(rv) or pd.isna(sd) or sd == 0:
        vwap_z = 0.0
    else:
        vwap_z = round(float((latest['close'] - rv) / sd), 2)
    score = 0.0
    if dev_atr > 0.5 and 50 < rsi < 70:
        score = min(1.0, (dev_atr / 3.0) * 0.5 + ((rsi - 50) / 20.0) * 0.5)
    elif dev_atr < -0.5 and 30 < rsi < 50:
        score = max(-1.0, (dev_atr / 3.0) * 0.5 + ((rsi - 50) / 20.0) * 0.5)
    return round(float(score), 2), round(float(rsi), 2), round(float(dev_atr), 2), vwap_z


def _parse_settle_time(value):
    try:
        dt = datetime.fromisoformat(str(value))
    except Exception:
        try:
            dt = datetime.strptime(str(value), '%Y-%m-%d %H:%M:%S')
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _atomic_write_csv(df):
    directory = os.path.dirname(os.path.abspath(CSV_FILE)) or '.'
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix='.quant_trades_tmp_', suffix='.csv')
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, CSV_FILE)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def settle_open_trades(exchange):
    if not os.path.exists(CSV_FILE):
        return 0
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"settlement read failed, skip this round: {e}")
        return 0
    if 'status' not in df.columns:
        return 0
    now = datetime.now(timezone.utc)
    settled = 0
    for idx, row in df[df['status'] == 'OPEN'].iterrows():
        settle_time = _parse_settle_time(row['settlement_time'])
        if settle_time is None:
            print(f"skip malformed settlement_time row {row.get('tx_id')}")
            continue
        if now < settle_time:
            continue
        try:
            ticker = exchange.fetch_ticker(row['symbol'])
            exit_price = ticker['last']
            if exit_price is None:
                continue
            if row['direction'] == 'LONG':
                pnl = (exit_price - row['entry_price']) / row['entry_price']
            else:
                pnl = (row['entry_price'] - exit_price) / row['entry_price']
            df.at[idx, 'exit_price'] = exit_price
            df.at[idx, 'pnl_pct'] = round(pnl * 100, 2)
            df.at[idx, 'status'] = 'CLOSED'
            settled += 1
            print(f"settled {row['symbol']} {row['direction']} pnl {df.at[idx, 'pnl_pct']}%")
        except Exception as e:
            print(f"settle {row['symbol']} failed, keep OPEN: {e}")
            continue
    if settled:
        _atomic_write_csv(df)
    return settled


def run_model():
    init_csv()
    exchange = ccxt.binance({'enableRateLimit': True})
    now_utc = datetime.now(timezone.utc)
    print(f"quant model run | {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    try:
        settled = settle_open_trades(exchange)
    except Exception as e:
        print(f"settlement step failed, continue scanning: {e}")
        settled = 0
    print(f"settled expired trades: {settled}")
    records_to_add = []
    skipped = []
    for sym in SYMBOLS:
        try:
            df = fetch_and_prep_data(exchange, sym)
            if len(df) < 30:
                print(f"{sym} closed candles too few ({len(df)}), skip")
                skipped.append(sym)
                continue
            close_price = float(df.iloc[-1]['close'])
            if close_price != close_price:
                skipped.append(sym)
                continue
            score, rsi, dev_atr, vwap_z = calculate_confluence_score(df)
            direction = None
            if score >= 0.75:
                direction = 'LONG'
            elif score <= -0.75:
                direction = 'SHORT'
            if direction:
                print(f"{sym} trigger {direction} closed close {close_price} score {score} RSI {rsi} DevATR {dev_atr} VWAP_Z {vwap_z}")
                tx_id = f"TX_{now_utc.strftime('%Y%m%d%H%M%S')}_{sym.replace('/', '')}"
                settle_time = now_utc + timedelta(hours=HOLDING_PERIOD_HOURS)
                records_to_add.append({
                    'tx_id': tx_id,
                    'timestamp': now_utc.isoformat(),
                    'symbol': sym,
                    'direction': direction,
                    'entry_price': close_price,
                    'rsi': rsi,
                    'dev_atr': dev_atr,
                    'vwap_z': vwap_z,
                    'score': score,
                    'settlement_time': settle_time.isoformat(),
                    'exit_price': '',
                    'pnl_pct': '',
                    'status': 'OPEN'
                })
        except Exception as e:
            print(f"{sym} failed, skip: {e}")
            skipped.append(sym)
    if records_to_add:
        add_df = pd.DataFrame(records_to_add)
        add_df.to_csv(CSV_FILE, mode='a', header=False, index=False)
        print(f"{len(records_to_add)} new signals appended to CSV.")
    else:
        print("FLAT, no trigger, no new records.")
    if skipped:
        print(f"skipped: {', '.join(skipped)}")


if __name__ == "__main__":
    run_model()
