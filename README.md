# 🧠 LLM do Zero (Build a Large Language Model from Scratch)

Este repositório contém a implementação e os estudos passo a passo baseados no livro **"Build a Large Language Model (From Scratch)"**.

O objetivo deste projeto é construir a arquitetura de um LLM desconstruindo cada etapa: desde o processamento inicial de texto (Tokenização) até a construção das camadas de Atenção e treinamento da Rede Neural utilizando PyTorch.

## 📁 Estrutura do Projeto
* `/src`: Códigos-fonte em Python (Tokenizer, validação de ambiente, etc).
* `/docs`: Documentação, glossários técnicos e respostas teóricas de cada capítulo.
* `/notebooks`: Arquivos de experimentação interativa.

## 🚀 Tecnologias e Bibliotecas
* Python 3.x
* PyTorch
* Expressões Regulares (re)

## 🛠️ Preparação do ambiente

Com Python 3 instalado, instale as dependências do projeto:

```bash
python -m pip install -r requirements.txt
```

O corpus de desenvolvimento da Sprint 2 é `the-verdict.txt`. Os caminhos e
parâmetros padrão dos experimentos ficam centralizados em `src/config.py`.

O escopo e a ordem de implementação da Sprint 2 estão documentados em
[`docs/sprint2.md`](docs/sprint2.md).
