```python
from coleta_dados import baixar_dados
from indicadores import adicionar_indicadores
from preprocessamento import preparar_dados
from modelo import criar_modelo
from sklearn.model_selection import train_test_split
from integracao_mt5 import conectar_mt5, obter_dados_mt5, enviar_ordem

# Conectar ao MetaTrader
conectar_mt5()

# Pegar os últimos 60 minutos e transformar em candle de 1h
df_min = obter_dados_mt5("PETR4", mt5.TIMEFRAME_M1, 60)
df = df_min.resample('1H').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'tick_volume':'sum'}).dropna()
df.rename(columns={'close': 'Close'}, inplace=True)

# ... segue o pipeline (adicionar_indicadores, preparar_dados etc.)
# Parâmetros
ticker = "PETR4.SA"
features = ['Close', 'rsi', 'sma5', 'sma10', 'macd', 'macd_signal']
janela = 3

# Pipeline
df = baixar_dados(ticker)
df = df.resample('1H').last().dropna()
df = adicionar_indicadores(df)

X, y, scaler = preparar_dados(df, features, janela)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = criar_modelo((janela, len(features)))
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

loss, acc = model.evaluate(X_test, y_test)
print(f"Acurácia: {acc * 100:.2f}%")

# Previsão
entrada = X[-1].reshape((1, janela, len(features)))
previsao = model.predict(entrada)[0][0]
print("Previsão próxima hora:", "Alta" if previsao > 0.5 else "Baixa")
```

from log_performance import registrar_log

# Depois de model.evaluate()
loss, acc = model.evaluate(X_test, y_test)

entrada = X[-1].reshape((1, janela, len(features)))
previsao = model.predict(entrada)[0][0]
resultado = "Alta" if previsao > 0.5 else "Baixa"
preco_atual = df['Close'].iloc[-1]

print(f"Acurácia: {acc * 100:.2f}%")
print("Previsão próxima hora:", resultado)

# Registrar no log
registrar_log(ticker, acc, resultado, preco_atual)
