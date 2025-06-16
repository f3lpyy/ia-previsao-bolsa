```python
import ta
import pandas as pd

def adicionar_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['sma5'] = df['Close'].rolling(window=5).mean()
    df['sma10'] = df['Close'].rolling(window=10).mean()
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    return df.dropna()
```
