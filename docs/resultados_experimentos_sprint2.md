# Resultados dos experimentos — Sprint 2

Os experimentos foram executados com `torch.manual_seed(123)`, usando o corpus `the-verdict.txt`, o tokenizador simples do projeto e CPU.

## 1. Estatísticas do corpus

| Métrica | Valor |
| --- | --- |
| Caracteres | 20479 |
| Tokens produzidos | 4690 |
| Tamanho do vocabulário | 1132 |
| IDs codificados | 4690 |
| Tokens especiais reservados | 2 |

## 2. Tamanho do contexto e quantidade de amostras

O stride foi mantido em 4 para permitir comparar diretamente o efeito do contexto. A quantidade de amostras segue `floor((N - (L + 1)) / S) + 1`.

| Context length | Stride | Amostras | Lotes (batch 4) |
| --- | --- | --- | --- |
| 4 | 4 | 1172 | 293 |
| 8 | 4 | 1171 | 293 |
| 16 | 4 | 1169 | 293 |
| 32 | 4 | 1165 | 292 |

## 3. Tamanho do lote

O contexto foi fixado em 8 e o stride em 4. `drop_last=False` foi usado para contabilizar também o último lote incompleto.

| Batch size | Lotes | Forma entrada | Forma alvo |
| --- | --- | --- | --- |
| 2 | 586 | (2, 8) | (2, 8) |
| 4 | 293 | (4, 8) | (4, 8) |
| 8 | 147 | (8, 8) | (8, 8) |
| 16 | 74 | (16, 8) | (16, 8) |

## 4. Dimensão dos embeddings

A entrada foi fixada em `batch_size=4` e `context_length=8`. O custo abaixo considera somente o tensor combinado em `float32`.

| Embedding dim | Forma combinada | Elementos | Memória |
| --- | --- | --- | --- |
| 16 | (4, 8, 16) | 512 | 2.00 KiB |
| 32 | (4, 8, 32) | 1024 | 4.00 KiB |
| 64 | (4, 8, 64) | 2048 | 8.00 KiB |
| 128 | (4, 8, 128) | 4096 | 16.00 KiB |

## 5. Textos e quantidade de tokens

| Texto | Tokens | UNK | Tokens produzidos |
| --- | --- | --- | --- |
| Hello, do you like tea? | 7 | 1 | ['Hello', ',', 'do', 'you', 'like', 'tea', '?'] |
| In the sunlit terraces of the palace. | 8 | 1 | ['In', 'the', 'sunlit', 'terraces', 'of', 'the', 'palace', '.'] |
| Jack Gisburn painted a portrait. | 6 | 0 | ['Jack', 'Gisburn', 'painted', 'a', 'portrait', '.'] |

## 6. Formas finais do pipeline

- Corpus codificado: `[1, 4690]` quando representado como uma sequência de IDs.
- Lote de entrada: `[4, 8]`.
- Lote de alvo: `[4, 8]`.
- Embedding combinado com dimensão 64: `[4, 8, 64]`.

## Reprodução

```powershell
.venv\Scripts\python.exe experiments\sprint2_experimentos.py
```
