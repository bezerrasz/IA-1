# Glossário Técnico - Capítulo 2

* **Token:** O bloco fundamental de texto que o modelo processa. Pode representar uma palavra, uma parte dela ou uma pontuação.
* **BPE (Byte-Pair Encoding):** Algoritmo de tokenização que funde pares de caracteres frequentes, criando um vocabulário dinâmico de subpalavras. É o padrão utilizado pela OpenAI.
* **<|unk|> (Unknown Token):** Token especial usado para representar palavras que estão fora do vocabulário (OOV) original do modelo.
* **<|endoftext|>:** Token especial que indica ao modelo que um texto ou documento acabou e que um novo assunto vai começar.
* **Sliding Window:** Método de formatação de dados de treinamento onde a sequência alvo (Target) é exatamente a sequência de entrada (Input) deslocada uma posição à direita.
* **Batch (Lote):** Um subconjunto de dados de treinamento agrupados para serem enviados à rede neural simultaneamente. 
* **Dataloader:** Classe do PyTorch responsável por embaralhar e organizar o dataset gigante em pequenos lotes (Batches) otimizados para a memória da GPU/CPU.
* **Vocabulary (Vocabulário / Dicionário):** O mapeamento definitivo que liga cada token único (string) a um ID (número inteiro) exclusivo.