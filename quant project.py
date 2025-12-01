# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import yfinance as yf
from backtesting import Backtest, Strategy

# =========================
# DOWNLOAD DATA
# =========================
data = yf.download("EURUSD=X", start="2019-01-01", interval="1d")

# Fix MultiIndex Columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [c[0] for c in data.columns]

# Keep OHLC only
data = data[['Open', 'High', 'Low', 'Close']].dropna()


# =========================
# INDICATORS (outside class)
# =========================

def ATR(h, l, c, n=14):
    hl = (h - l)
    hpc = np.abs(h - np.roll(c, 1))
    lpc = np.abs(l - np.roll(c, 1))
    tr = np.maximum.reduce([hl, hpc, lpc])
    tr[0] = hl[0]
    return pd.Series(tr).rolling(n).mean()


def RSI(c, n=14):
    delta = np.diff(c, prepend=c[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = pd.Series(gain).rolling(n).mean()
    avg_loss = pd.Series(loss).rolling(n).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def BB(c, n=20, k=2):
    mid = pd.Series(c).rolling(n).mean()
    std = pd.Series(c).rolling(n).std()
    return mid, mid + k * std, mid - k * std


def TREND(c, n=50):
    return pd.Series(c).rolling(n).mean()


# Volatility Regime using ATR Z-score
def REGIME(atr):
    z = (atr - pd.Series(atr).rolling(100).mean()) / pd.Series(atr).rolling(100).std()
    z = z.fillna(0)

    out = np.zeros(len(z))  # LOW VOL
    out[z > 0.4] = 1  # MED VOL
    out[z > 1.2] = 2  # HIGH VOL
    return out


# =========================
# STRATEGY
# =========================
class VolatilityAdaptiveFX(Strategy):

    def init(self):
        c = self.data.Close
        h = self.data.High
        l = self.data.Low

        self.atr = self.I(ATR, h, l, c)
        self.rsi = self.I(RSI, c)
        self.bb_mid, self.bb_up, self.bb_low = self.I(BB, c)
        self.trend = self.I(TREND, c)
        self.regime = self.I(REGIME, self.atr)

    def next(self):
        price = self.data.Close[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        regime = self.regime[-1]

        # 🔻 EXIT LOGIC
        if self.position:
            if self.position.is_long and rsi > 60:
                return self.position.close()
            if self.position.is_short and rsi < 40:
                return self.position.close()

        # 🔥 ENTRY LOGIC
        # LOW VOL (mean reversion)
        if regime == 0:
            if price < self.bb_low[-1] and rsi < 35:
                self.buy(sl=price - 2 * atr, tp=price + 3 * atr)
            if price > self.bb_up[-1] and rsi > 65:
                self.sell(sl=price + 2 * atr, tp=price - 3 * atr)

        # MED VOL (breakouts)
        elif regime == 1:
            if price > self.bb_up[-1]:
                self.buy(sl=price - 2 * atr, tp=price + 3 * atr)
            if price < self.bb_low[-1]:
                self.sell(sl=price + 2 * atr, tp=price - 3 * atr)

        # HIGH VOL (trend following)
        elif regime == 2:
            if price > self.trend[-1]:
                self.buy(sl=price - 2 * atr, tp=price + 3 * atr)
            if price < self.trend[-1]:
                self.sell(sl=price + 2 * atr, tp=price - 3 * atr)


# =========================
# RUN BACKTEST (FINAL)
# =========================
bt = Backtest(
    data,
    VolatilityAdaptiveFX,
    cash=10000,
    commission=0.0002,
    trade_on_close=True
)

stats = bt.run()
print(stats)

bt.plot(filename="equity_curve.html", open_browser=True)
print("\nSaved → equity_curve.html ")