# 🧠 LLM Tokenizer from Scratch

Este repositório contém a implementação passo a passo de um Tokenizer construído do zero, baseado nos ensinamentos do livro **"Build a Large Language Model (From Scratch)"**.

O objetivo deste projeto é entender a fundo a matemática e a lógica de engenharia por trás do processamento de linguagem natural (NLP), desconstruindo textos em tokens computáveis antes de alimentá-los em uma rede neural.

## 🛠️ O que foi implementado
* **Construção de Vocabulário:** Leitura de um texto base (`the-verdict.txt`) para mapeamento e extração de tokens únicos através de Expressões Regulares (Regex).
* **Conversão Bidirecional (Encode / Decode):** 
  * `encode`: Transforma a linguagem humana (strings) em matrizes de IDs (inteiros).
  * `decode`: Converte o processamento da máquina (IDs) de volta para linguagem humana, mantendo a formatação e pontuação corretas.
* **Prevenção de OOV (Out-Of-Vocabulary):** Implementação de token especial `<|unk|>` para evitar falhas (`KeyError`) quando o algoritmo esbarra em palavras desconhecidas fora de sua base de treinamento inicial.

## 🚀 Como executar

1. Clone o repositório:
```bash
git clone [https://github.com/bezerrasz/IA-1.git](https://github.com/bezerrasz/IA-1.git)