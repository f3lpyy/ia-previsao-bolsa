import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

def conectar_mt5():
    if not mt5.initialize():
        raise Exception("Erro ao conectar ao MetaTrader 5")

def obter_dados_mt5(symbol="PETR4", timeframe=mt5.TIMEFRAME_M1, n=60):
    ticks = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    df = pd.DataFrame(ticks)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def enviar_ordem(symbol, tipo="compra", volume=1.0):
    tipo_ordem = mt5.ORDER_TYPE_BUY if tipo == "compra" else mt5.ORDER_TYPE_SELL
    preco = mt5.symbol_info_tick(symbol).ask if tipo == "compra" else mt5.symbol_info_tick(symbol).bid
    
    ordem = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": tipo_ordem,
        "price": preco,
        "deviation": 10,
        "magic": 123456,
        "comment": "IA previsão bolsa",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    resultado = mt5.order_send(ordem)
    print(f"Ordem enviada: {resultado}")
