import os
import re
import requests

# IMPORTAÇÃO/DOWNLOAD DO TEXTO 
if not os.path.exists("the-verdict.txt"):
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    file_path = "the-verdict.txt"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)

# ABERTURA DO ARQUIVO 
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# NUMERO TOTAL DE CARACTERES E PRIMEIROS 100 CARACTERES DO TEXTO 
print("Total number of character:", len(raw_text))
print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])

print(len(preprocessed))

# TAMANHO DO VOCABULÁRIO
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)

# TRANSFORMAÇÃO DE PALAVRAS EM INTEIROS E MOSTRANDO OS 50 PRIMEIROS ITENS 
vocab = {token:integer for integer,token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break



