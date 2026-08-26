# 🧠 LLM do Zero (Build a Large Language Model from Scratch)

Este repositório contém a implementação e os estudos passo a passo baseados no livro **"Build a Large Language Model (From Scratch)"**.

O objetivo deste projeto é construir a arquitetura de um LLM desconstruindo cada etapa: desde o processamento inicial de texto (Tokenização) até a construção das camadas de Atenção e treinamento da Rede Neural utilizando PyTorch.

## 📁 Estrutura do Projeto
* `/src`: Códigos-fonte em Python (Tokenizer, validação de ambiente, etc).
* `/docs`: Documentação, glossários técnicos e respostas teóricas de cada capítulo.
* `/notebooks`: Arquivos de experimentação interativa.
* `/experiments`: Scripts reproduzíveis e resultados dos experimentos.
* `/tests`: Testes automatizados dos componentes.

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

## ▶️ Execução da Sprint 2

Na primeira execução, crie um ambiente virtual e instale as dependências:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Execute os testes automatizados:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Execute o experimento completo e gere o relatório:

```powershell
.venv\Scripts\python.exe experiments\sprint2_experimentos.py
```

O notebook de demonstração do pipeline pode ser executado com:

```powershell
.venv\Scripts\python.exe notebooks\notebook.py
```

## ✅ Entregáveis da Sprint 2

- Leitura orientada do Capítulo 2 em `docs/leitura_orientada_capitulo2.md`;
- Glossário técnico atualizado em `docs/glossario_capitulo2.md`;
- Tokenização, vocabulário e Token IDs em `src/tokenizer.py`;
- Sequências de treinamento em `src/dataset.py`;
- Embeddings e Positional Embeddings em `src/embeddings.py`;
- Dataset e DataLoader em `src/dataloader.py`;
- Experimentos e resultados em `experiments/` e `docs/resultados_experimentos_sprint2.md`;
- Análise técnica em `docs/analise_resultados_sprint2.md`;
- Compatibilidade preservada pelo módulo legado `src/tokeniz.py`.
