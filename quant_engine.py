import ccxt
import pandas as pd
import ta
import os
import tempfile
import json
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

CSV_FILE = 'quant_trades.csv'
SUMMARY_FILE = 'summary.csv'
HOLDING_PERIOD_HOURS = 24
STRATEGY_TYPE = 'atr_confluence'
SOURCE_POOL = 'binance_1h'
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
HEADER = "tx_id,timestamp,symbol,direction,entry_price,stop_loss,take_profit,rsi,dev_atr,vwap_z,score,settlement_time,settled_at,exit_price,exit_reason,pnl_pct,is_win,status,strategy_type,source_pool\n"

STABLECOIN_IDS = {
    'tether', 'usd-coin', 'dai', 'first-digital-usd', 'true-usd',
    'paypal-usd', 'usd1-wlfi', 'ethena-usde', 'usdd', 'frax',
    'pax-dollar', 'gemini-dollar', 'stasis-euro', 'euro-coin',
}
STABLECOIN_SYMBOLS = {'usdt', 'usdc', 'dai', 'fdusd', 'tusd', 'pyusd', 'usd1', 'usde', 'usdd', 'frax', 'usdp', 'gusd', 'eurc'}

FALLBACK_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'SOL/USDT',
    'DOGE/USDT', 'ADA/USDT', 'TRX/USDT', 'AVAX/USDT', 'LINK/USDT',
    'DOT/USDT', 'LTC/USDT', 'BCH/USDT', 'NEAR/USDT', 'UNI/USDT',
    'APT/USDT', 'ATOM/USDT', 'ETC/USDT', 'XLM/USDT', 'SUI/USDT',
]


def get_top_symbols():
    url = 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=50&page=1&sparkline=false'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'quant-trading-engine/1.0', 'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        symbols = []
        for item in data:
            coin_id = str(item.get('id', '')).lower()
            sym = str(item.get('symbol', '')).strip().upper()
            if not sym or coin_id in STABLECOIN_IDS or sym.lower() in STABLECOIN_SYMBOLS:
                continue
            pair = f"{sym}/USDT"
            if pair not in symbols:
                symbols.append(pair)
        if len(symbols) >= 10:
            print(f"Top50 universe from CoinGecko: {len(symbols)} pairs after stablecoin filter")
            return symbols
        print(f"CoinGecko returned only {len(symbols)} pairs, using fallback list")
    except Exception as e:
        print(f"CoinGecko top50 fetch failed, using fallback list: {e}")
    return list(FALLBACK_SYMBOLS)


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


def _parse_time(value):
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


def _atomic_write_csv(df, path):
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix='.quant_tmp_', suffix='.csv')
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def _get_float(row, col):
    try:
        v = row.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if isinstance(v, str) and v.strip() == '':
            return None
        return float(v)
    except Exception:
        return None


def _close_trade(df, idx, exit_price, exit_reason, settled_at):
    row = df.loc[idx]
    entry = float(row['entry_price'])
    if row['direction'] == 'LONG':
        pnl = (exit_price - entry) / entry * 100
    else:
        pnl = (entry - exit_price) / entry * 100
    df.at[idx, 'exit_price'] = exit_price
    df.at[idx, 'exit_reason'] = exit_reason
    df.at[idx, 'pnl_pct'] = round(float(pnl), 2)
    df.at[idx, 'is_win'] = 1 if pnl > 0 else 0
    df.at[idx, 'settled_at'] = settled_at
    df.at[idx, 'status'] = 'CLOSED'
    return float(pnl)


def settle_open_trades(exchange, market_data):
    if not os.path.exists(CSV_FILE):
        return 0
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"settlement read failed, skip this round: {e}")
        return 0
    if 'status' not in df.columns or df.empty:
        return 0
    for col in ['exit_reason', 'settled_at', 'status']:
        if col in df.columns:
            df[col] = df[col].astype(object)
    now = datetime.now(timezone.utc)
    settled = 0
    for idx, row in df[df['status'] == 'OPEN'].iterrows():
        try:
            entry_dt = _parse_time(row['timestamp'])
            settle_dt = _parse_time(row['settlement_time'])
            if entry_dt is None or settle_dt is None:
                print(f"skip malformed time row {row.get('tx_id')}")
                continue
            symbol = row['symbol']
            direction = row['direction']
            entry = float(row['entry_price'])
            stop_loss = _get_float(row, 'stop_loss')
            take_profit = _get_float(row, 'take_profit')
            closed = False
            data = market_data.get(symbol) if market_data else None
            if data is not None and 'df' in data:
                cdf = data['df']
                for _, candle in cdf.iterrows():
                    try:
                        candle_dt = datetime.fromtimestamp(float(candle['timestamp']) / 1000, tz=timezone.utc)
                    except Exception:
                        continue
                    if candle_dt <= entry_dt:
                        continue
                    candle_close_dt = candle_dt + timedelta(hours=1)
                    low = float(candle['low'])
                    high = float(candle['high'])
                    if direction == 'LONG':
                        if stop_loss is not None and low <= stop_loss:
                            _close_trade(df, idx, stop_loss, 'stop_loss', candle_close_dt.isoformat())
                            closed = True
                            break
                        if take_profit is not None and high >= take_profit:
                            _close_trade(df, idx, take_profit, 'take_profit', candle_close_dt.isoformat())
                            closed = True
                            break
                    else:
                        if stop_loss is not None and high >= stop_loss:
                            _close_trade(df, idx, stop_loss, 'stop_loss', candle_close_dt.isoformat())
                            closed = True
                            break
                        if take_profit is not None and low <= take_profit:
                            _close_trade(df, idx, take_profit, 'take_profit', candle_close_dt.isoformat())
                            closed = True
                            break
            if closed:
                settled += 1
                print(f"settled {symbol} {direction} by SL/TP pnl {df.at[idx, 'pnl_pct']}% reason {df.at[idx, 'exit_reason']}")
                continue
            if now < settle_dt:
                continue
            ticker = None
            if data is not None and 'live_price' in data:
                ticker = data['live_price']
            else:
                try:
                    ticker = exchange.fetch_ticker(symbol)['last']
                except Exception as e:
                    print(f"settle {symbol} ticker failed, keep OPEN: {e}")
                    continue
            if ticker is None:
                continue
            if direction == 'LONG':
                if stop_loss is not None and ticker <= stop_loss:
                    _close_trade(df, idx, stop_loss, 'stop_loss', now.isoformat())
                elif take_profit is not None and ticker >= take_profit:
                    _close_trade(df, idx, take_profit, 'take_profit', now.isoformat())
                else:
                    _close_trade(df, idx, ticker, 'time_expiry', now.isoformat())
            else:
                if stop_loss is not None and ticker >= stop_loss:
                    _close_trade(df, idx, stop_loss, 'stop_loss', now.isoformat())
                elif take_profit is not None and ticker <= take_profit:
                    _close_trade(df, idx, take_profit, 'take_profit', now.isoformat())
                else:
                    _close_trade(df, idx, ticker, 'time_expiry', now.isoformat())
            settled += 1
            print(f"settled {symbol} {direction} at expiry pnl {df.at[idx, 'pnl_pct']}% reason {df.at[idx, 'exit_reason']}")
        except Exception as e:
            print(f"settle row {row.get('tx_id')} failed, keep OPEN: {e}")
            continue
    if settled:
        _atomic_write_csv(df, CSV_FILE)
    return settled


def _summary_rows(df_closed):
    def stats(sub):
        if sub.empty:
            return None
        pnl = pd.to_numeric(sub['pnl_pct'], errors='coerce').dropna()
        if pnl.empty:
            return None
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        win_sum = float(wins.sum()) if not wins.empty else 0.0
        loss_sum = float(losses.sum()) if not losses.empty else 0.0
        profit_factor = round(win_sum / abs(loss_sum), 2) if loss_sum != 0 else 0.0
        cum = pnl.cumsum()
        peak = cum.cummax()
        drawdown = float((peak - cum).max()) if not cum.empty else 0.0
        return {
            'total_trades': int(len(pnl)),
            'win_rate': round(float((pnl > 0).mean()), 4),
            'avg_win_pct': round(float(wins.mean()), 2) if not wins.empty else 0.0,
            'avg_loss_pct': round(float(losses.mean()), 2) if not losses.empty else 0.0,
            'profit_factor': profit_factor,
            'expectancy_pct': round(float(pnl.mean()), 2),
            'max_drawdown_pct': round(drawdown, 2),
        }
    rows = []
    overall = stats(df_closed)
    if overall:
        rows.append({'scope': 'overall', 'symbol': '', **overall})
    for sym, sub in df_closed.groupby('symbol'):
        s = stats(sub)
        if s:
            rows.append({'scope': 'symbol', 'symbol': sym, **s})
    return rows


SUMMARY_COLUMNS = ['scope', 'symbol', 'total_trades', 'win_rate', 'avg_win_pct', 'avg_loss_pct', 'profit_factor', 'expectancy_pct', 'max_drawdown_pct']


def write_summary():
    # Always write summary.csv, even with no closed trades yet, so the
    # workflow's git add never fails on a missing file (first-run deadlock).
    if not os.path.exists(CSV_FILE):
        pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(SUMMARY_FILE, index=False)
        return []
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"summary read failed: {e}")
        pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(SUMMARY_FILE, index=False)
        return []
    if 'status' not in df.columns or df.empty:
        pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(SUMMARY_FILE, index=False)
        return []
    closed = df[df['status'] == 'CLOSED'].copy()
    if closed.empty:
        pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(SUMMARY_FILE, index=False)
        print("summary: no closed trades yet, header-only summary.csv written")
        return []
    if 'settled_at' in closed.columns:
        closed = closed.sort_values('settled_at', na_position='last')
    rows = _summary_rows(closed)
    out = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    out.to_csv(SUMMARY_FILE, index=False)
    print("summary:")
    print(out.to_string(index=False))
    return rows


def run_model():
    init_csv()
    exchange = ccxt.binance({'enableRateLimit': True})
    now_utc = datetime.now(timezone.utc)
    print(f"quant model run | {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    symbols = get_top_symbols()
    print(f"universe: {len(symbols)} symbols")
    market_data = {}
    skipped = []
    for sym in symbols:
        try:
            df = fetch_and_prep_data(exchange, sym)
            if len(df) < 30:
                print(f"{sym} closed candles too few ({len(df)}), skip")
                skipped.append(sym)
                continue
            try:
                live_price = exchange.fetch_ticker(sym)['last']
            except Exception:
                live_price = float(df.iloc[-1]['close'])
            market_data[sym] = {'df': df, 'live_price': live_price}
        except Exception as e:
            print(f"{sym} failed, skip: {e}")
            skipped.append(sym)
    try:
        settled = settle_open_trades(exchange, market_data)
    except Exception as e:
        print(f"settlement step failed, continue scanning: {e}")
        settled = 0
    print(f"settled expired trades: {settled}")
    open_symbols = set()
    try:
        existing = pd.read_csv(CSV_FILE)
        if 'status' in existing.columns:
            open_symbols = set(existing[existing['status'] == 'OPEN']['symbol'].tolist())
    except Exception:
        open_symbols = set()
    records_to_add = []
    for sym, data in market_data.items():
        try:
            df = data['df']
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
            if not direction:
                continue
            if sym in open_symbols:
                print(f"{sym} already has OPEN trade, skip duplicate signal")
                continue
            atr = float(df.iloc[-1]['atr'])
            if direction == 'LONG':
                stop_loss = round(close_price - SL_ATR_MULT * atr, 8)
                take_profit = round(close_price + TP_ATR_MULT * atr, 8)
            else:
                stop_loss = round(close_price + SL_ATR_MULT * atr, 8)
                take_profit = round(close_price - TP_ATR_MULT * atr, 8)
            print(f"{sym} trigger {direction} closed close {close_price} score {score} RSI {rsi} DevATR {dev_atr} VWAP_Z {vwap_z} SL {stop_loss} TP {take_profit}")
            tx_id = f"TX_{now_utc.strftime('%Y%m%d%H%M%S')}_{sym.replace('/', '')}"
            settle_time = now_utc + timedelta(hours=HOLDING_PERIOD_HOURS)
            records_to_add.append({
                'tx_id': tx_id,
                'timestamp': now_utc.isoformat(),
                'symbol': sym,
                'direction': direction,
                'entry_price': close_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'rsi': rsi,
                'dev_atr': dev_atr,
                'vwap_z': vwap_z,
                'score': score,
                'settlement_time': settle_time.isoformat(),
                'settled_at': '',
                'exit_price': '',
                'exit_reason': '',
                'pnl_pct': '',
                'is_win': '',
                'status': 'OPEN',
                'strategy_type': STRATEGY_TYPE,
                'source_pool': SOURCE_POOL,
            })
        except Exception as e:
            print(f"{sym} scoring failed, skip: {e}")
            skipped.append(sym)
    if records_to_add:
        add_df = pd.DataFrame(records_to_add)
        add_df.to_csv(CSV_FILE, mode='a', header=False, index=False)
        print(f"{len(records_to_add)} new signals appended to CSV.")
    else:
        print("FLAT, no new records.")
    if skipped:
        print(f"skipped: {', '.join(sorted(set(skipped)))}")
    write_summary()


if __name__ == "__main__":
    run_model()
