# 🧠 LLM do Zero (Build a Large Language Model from Scratch)

Este repositório contém a implementação e os estudos passo a passo baseados no livro **"Build a Large Language Model (From Scratch)"** de Sebastian Raschka.

O objetivo deste projeto é construir a arquitetura de uma LLM desconstruindo cada etapa: desde o processamento inicial de texto (Tokenização) até a construção das camadas de Atenção e treinamento da Rede Neural utilizando PyTorch.

## 📁 Estrutura do Projeto

A arquitetura do projeto foi simplificada para centralizar a lógica e facilitar a visualização dos experimentos teóricos:

* `/src`: Motores lógicos e arquitetura do modelo.
  * `theverdict.py`: Download e validação do corpus de treinamento.
  * `tokeniz.py`: Processamento de texto, vocabulário, classes do Tokenizer e DataLoader (Capítulo 2).
  * `attention.py`: Implementação matemática das classes de Self-Attention, Causal Attention e Multi-Head Attention (Capítulo 3).
* `/notebooks`: Ambiente interativo de testes.
  * `notebook.py`: Script central que importa os módulos do `src` e gera os relatórios de execução passo a passo.
* `/docs`: Documentação técnica e conceitos teóricos.
  * Documentos e glossários divididos por capítulo (Capítulo 1, Capítulo 2 e Capítulo 3).

## 🚀 Tecnologias e Bibliotecas

* Python 3.x
* PyTorch (`torch`)
* Requests (`requests`)
* Expressões Regulares (`re`)

## 🛠️ Preparação do Ambiente

Para rodar o projeto do zero, crie um ambiente virtual e instale as dependências básicas:

```bash
# Criação do ambiente virtual
python -m venv .venv

# Ativação (Windows)
.\.venv\Scripts\activate

# Instalação das bibliotecas necessárias
pip install torch requests