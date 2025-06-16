```python
import yfinance as yf

def baixar_dados(ticker="PETR4.SA", periodo="30d", intervalo="1m"):
    data = yf.download(ticker, period=periodo, interval=intervalo)
    return data
```
