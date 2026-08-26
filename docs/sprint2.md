# Sprint 2 — Organização e escopo

## Objetivo

Construir o pipeline que transforma texto em lotes numéricos prontos para
serem consumidos pelo modelo:

```text
texto → tokens → Token IDs → sequências → embeddings
       → positional embeddings → lotes de dados
```

## Base definida nesta etapa

- Corpus inicial: `the-verdict.txt`.
- Caminho do corpus: centralizado em `src/config.py`.
- Tokens especiais: `<|endoftext|>` e `<|unk|>`.
- Parâmetros padrão dos experimentos: `DEFAULT_CONFIG` em `src/config.py`.
- Dependências: declaradas em `requirements.txt`.

## Ordem de implementação

1. Consolidar tokenização e vocabulário.
2. Validar conversão token ↔ Token ID.
3. Criar janelas de contexto e alvos deslocados.
4. Implementar embeddings e positional embeddings.
5. Organizar `Dataset` e `DataLoader`.
6. Executar experimentos com contexto, lote e dimensão de embedding variáveis.
7. Registrar resultados e análise técnica.

## Critérios de validação

O pipeline final deverá produzir, para um lote, entradas e alvos com forma
`[batch_size, context_length]`, e embeddings combinados com forma
`[batch_size, context_length, embedding_dim]`.

O mecanismo de atenção não faz parte desta Sprint; ele consumirá os embeddings
com informação posicional produzidos aqui na Sprint seguinte.
