📁 ia_previsao_bolsa/
├── 📄 requirements.txt
├── 📄 main.py
├── 📄 indicadores.py
├── 📄 modelo.py
├── 📄 coleta_dados.py
├── 📄 preprocessamento.py
└── 📄 README.md

---

### 📄 requirements.txt
```txt
pandas
yfinance
numpy
scikit-learn
tensorflow
keras
ta
```

---

### 📄 coleta_dados.py
```python
import yfinance as yf

def baixar_dados(ticker="PETR4.SA", periodo="30d", intervalo="1m"):
    data = yf.download(ticker, period=periodo, interval=intervalo)
    return data
```

---

### 📄 indicadores.py
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

---

### 📄 preprocessamento.py
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preparar_dados(df: pd.DataFrame, features, janela=3):
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    
    X = df[features].values
    y = df['target'].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - janela):
        X_seq.append(X_scaled[i:i+janela])
        y_seq.append(y[i+janela])

    return np.array(X_seq), np.array(y_seq), scaler
```

---

### 📄 modelo.py
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def criar_modelo(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

---

### 📄 main.py
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

---

### 📄 README.md
```markdown
# IA de Previsão de Bolsa de Valores

Este projeto é uma Inteligência Artificial que utiliza aprendizado de máquina (LSTM) para prever se o preço de uma ação subirá ou cairá na próxima hora.

## Funcionalidades
- Coleta dados de ações em tempo real (via Yahoo Finance)
- Calcula indicadores técnicos (RSI, SMA, MACD)
- Treina modelo de rede neural recorrente (LSTM)
- Previsão de tendência (alta/baixa)

## Como usar
```bash
pip install -r requirements.txt
python main.py
```

## Requisitos
- Python 3.8+
- Conexão com a internet

## Aviso
Este projeto é educativo. Resultados reais podem variar e **não é recomendada a utilização para operações financeiras reais sem testes rigorosos**.
```
