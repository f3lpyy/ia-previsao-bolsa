import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Título
st.title("📊 BörsaMind - Dashboard de Performance")

# Ler o log
log_path = "logs_desempenho.csv"
try:
    df = pd.read_csv(log_path)
except FileNotFoundError:
    st.warning("Nenhum dado encontrado. Execute o robô primeiro para gerar o log.")
    st.stop()

# Converter data
df['DataHora'] = pd.to_datetime(df['DataHora'])

# Mostrar dados brutos
with st.expander("📂 Ver dados brutos"):
    st.dataframe(df.tail(20))

# Gráfico de acurácia ao longo do tempo
st.subheader("📈 Evolução da Acurácia")
fig, ax = plt.subplots()
ax.plot(df['DataHora'], df['Acuracia'].astype(float), marker='o', linestyle='-')
ax.set_xlabel("Data e Hora")
ax.set_ylabel("Acurácia")
ax.set_ylim(0, 1)
st.pyplot(fig)

# Estatísticas rápidas
st.subheader("📌 Resumo")
col1, col2 = st.columns(2)
with col1:
    st.metric("Última previsão", df['Previsao'].iloc[-1])
    st.metric("Última acurácia (%)", f"{float(df['Acuracia'].iloc[-1]) * 100:.2f}")
with col2:
    st.metric("Execuções totais", len(df))
    st.metric("Ativo", df['Ativo'].iloc[-1])
