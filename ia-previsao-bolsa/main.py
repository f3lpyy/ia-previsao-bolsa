```python
from coleta_dados import baixar_dados
from indicadores import adicionar_indicadores
from preprocessamento import preparar_dados
from modelo import criar_modelo
from sklearn.model_selection import train_test_split

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
