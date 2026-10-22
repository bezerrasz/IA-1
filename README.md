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

## Treinar o GPT e usar os pesos salvos

Para executar um treinamento inicial pequeno na CPU:

```powershell
.\.venv\Scripts\python.exe experiments\treinar_gpt.py
```

O programa separa treino/validação, registra as perdas e salva o melhor modelo
em `experiments/gpt_treinado.pt`. Depois, teste uma continuação ou digite frases
no modo interativo:

```powershell
.\.venv\Scripts\python.exe experiments\gerar_texto.py --prompt "I HAD always"
.\.venv\Scripts\python.exe experiments\gerar_texto.py --interactive
```

No modo interativo, `/sair` encerra a execução. O modelo aprende com um conto
curto em inglês e gera continuações; sua qualidade é limitada pelo tamanho dos
dados e da arquitetura. Os experimentos originais da Sprint 4 continuam usando
pesos aleatórios para comparação; `gerar_texto.py` carrega os pesos treinados.

O procedimento, as opções e as limitações estão em
[`docs/treinamento.md`](docs/treinamento.md). O comando de treino gera o relatório
`docs/resultados_treinamento.md`. Os pesos binários e o JSON de métricas ficam
locais, fora do versionamento.

## Usar e treinar em português

Para usar o checkpoint em português no modo interativo:

```powershell
.\.venv\Scripts\python.exe experiments\gerar_texto.py --language pt --interactive
```

Na primeira execução em outra máquina, produza os pesos com:

```powershell
.\.venv\Scripts\python.exe experiments\treinar_gpt.py --language pt
```

O perfil revisado usa `corpus_dialogos_pt_treino.txt`, um corpus sintético
didático de perguntas e respostas, e salva `experiments/gpt_portugues_v2.pt` sem substituir o modelo em
inglês. Experimente `olá, tudo bem?` ou `O que você faz?`; essas perguntas estão
nos dados de treinamento. A geração é feita pelo GPT e continua sujeita a erros
e repetições. `/sair` encerra o modo interativo.

Veja o procedimento, a origem dos dados e as limitações em
[`docs/portugues.md`](docs/portugues.md). O treino gera
`docs/resultados_treinamento_portugues_v2.md`. Os pesos ficam locais, fora do Git.

Para avaliar as respostas do modelo treinado, além dos testes de código:

```powershell
.\.venv\Scripts\python.exe experiments\avaliar_portugues.py
```

O relatório `docs/avaliacao_portugues.md` separa as perguntas já vistas no treino
das variações de um teste controlado. Esse experimento não mede conversa geral.

## ✅ Integração da Sprint 3

As classes didáticas da Sprint 3 permanecem disponíveis em
`src/attention.py`: `SelfAttention_v1`, `SelfAttention_v2`,
`CausalAttention` e `MultiHeadAttention`. A Multi-Head Attention usada pelo
GPT mantém a assinatura compatível com os experimentos anteriores e adiciona
validação de dimensões, máscara causal e suporte ao Transformer Block.

O glossário correspondente está em `docs/glossario_capitulo3.md`.

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
