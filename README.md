# 🧠 LLM do Zero (Build a Large Language Model from Scratch)

Este repositório contém a implementação e os estudos passo a passo baseados no livro **"Build a Large Language Model (From Scratch)"**.

O objetivo deste projeto é construir a arquitetura de um LLM desconstruindo cada etapa: desde o processamento inicial de texto (Tokenização) até a construção das camadas de Atenção e treinamento da Rede Neural utilizando PyTorch.

## 📁 Estrutura do Projeto
* `/src`: Códigos-fonte em Python (Tokenizer, embeddings, atenção, Transformer e GPT).
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

## ▶️ Execução da Sprint 4

Execute os testes, incluindo atenção, Transformer Block, GPT e geração:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Execute os experimentos de arquitetura e geração. O comando reproduz os
resultados em [`docs/resultados_experimentos_sprint4.md`](docs/resultados_experimentos_sprint4.md):

```powershell
.venv\Scripts\python.exe experiments\sprint4_experimentos.py
```

Os componentes da Sprint 4 podem ser usados diretamente assim:

```python
import torch
from src.gpt import GPTModel

model = GPTModel(
    vocab_size=1132,
    context_length=16,
    embedding_dim=32,
    num_heads=4,
    num_layers=2,
    feed_forward_dim=64,
)
logits = model(torch.tensor([[1, 2, 3, 4]]))  # [1, 4, 1132]
generated_ids = model.generate(
    torch.tensor([[1, 2, 3, 4]]),
    max_new_tokens=4,
    temperature=0.8,
    top_k=10,
    do_sample=True,
)
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

## ✅ Entregáveis da Sprint 4

- Glossário do Capítulo 4 em `docs/glossario_capitulo4.md`;
- Multi-Head Attention causal e Transformer Block em `src/attention.py` e
  `src/transformer.py`;
- Arquitetura GPT, LayerNorm final e Language Model Head em `src/gpt.py`;
- Inferência, greedy decoding, temperatura e top-k em `src/gpt.py` e
  `src/generation.py`;
- Documentação dos componentes em `docs/transformer_block.md`,
  `docs/arquitetura_gpt.md` e `docs/inferencia_geracao.md`;
- Experimentos reproduzíveis em `experiments/sprint4_experimentos.py`, com
  resultados e análise em `docs/resultados_experimentos_sprint4.md` e
  `docs/analise_resultados_sprint4.md`;
- Testes automatizados de atenção, Transformer, GPT e geração em `tests/`.
