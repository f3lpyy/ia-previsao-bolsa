import csv
from datetime import datetime
import os

def registrar_log(ativo, acuracia, previsao, preco_atual):
    nome_arquivo = "logs_desempenho.csv"
    existe = os.path.exists(nome_arquivo)

    with open(nome_arquivo, mode="a", newline="") as arquivo:
        writer = csv.writer(arquivo)
        if not existe:
            writer.writerow(["DataHora", "Ativo", "Acuracia", "Previsao", "PrecoAtual"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ativo,
            f"{acuracia:.4f}",
            previsao,
            f"{preco_atual:.2f}"
        ])
