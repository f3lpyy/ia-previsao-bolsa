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