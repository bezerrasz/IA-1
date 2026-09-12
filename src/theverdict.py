import os
import requests

# Garante que o arquivo será salvo/lido na raiz do projeto
diretorio_atual = os.path.dirname(__file__)
file_path = os.path.join(diretorio_atual, "..", "the-verdict.txt")

# IMPORTAÇÃO/DOWNLOAD DO TEXTO 
if not os.path.exists(file_path):
    print("Baixando o arquivo the-verdict.txt...")
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    with open(file_path, "wb") as f:
        f.write(response.content)
    print("Download concluído com sucesso!")
else:
    print("Tudo certo! O arquivo the-verdict.txt já está na raiz do projeto.")