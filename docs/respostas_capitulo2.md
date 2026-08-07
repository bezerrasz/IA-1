# Respostas Teóricas - Capítulo 2 (Working with Text Data)

**1. O que é tokenização e por que ela é uma etapa obrigatória?**
A tokenização é o processo de quebrar um texto cru em partes menores (tokens), que podem ser palavras, subpalavras ou caracteres. Ela é obrigatória porque redes neurais não conseguem processar letras ou strings nativamente; elas realizam cálculos matemáticos. A tokenização é a ponte para converter texto legível em IDs numéricos (inteiros).

**2. Como o modelo lida com palavras que não aprendeu (Out-Of-Vocabulary - OOV)?**
Em tokenizadores simples, palavras não mapeadas no dicionário base geram erros. Para resolver isso, injetamos tokens especiais no vocabulário, como o `<|unk|>` (Unknown). Quando o modelo esbarra em uma palavra nova, ele a substitui pelo ID do `<|unk|>`, permitindo que o processamento continue sem travar.

**3. O que é o algoritmo BPE (Byte-Pair Encoding)?**
É um método avançado de tokenização de subpalavras usado no GPT. Em vez de registrar palavras inteiras (o que geraria um dicionário infinito) ou letras soltas (que perdem o sentido semântico), o BPE agrupa as combinações de caracteres que mais se repetem no texto. Isso permite que o modelo entenda palavras desconhecidas quebrando-as em sílabas/partes conhecidas.

**4. O que é a técnica de "Sliding Window" (Janela Deslizante) na criação do dataset?**
É a forma como ensinamos o modelo a prever a próxima palavra. Criamos pares de dados (X e Y) deslizando uma janela sobre o texto: se a entrada (X) tem 4 tokens, o alvo (Y) será a mesma sequência deslocada uma posição para a direita. Exemplo: Se X é "Eu gosto de tomar", Y será "gosto de tomar café".

**5. Qual a função do DataLoader e dos Batches (lotes)?**
Treinar um modelo com um livro inteiro de uma vez estouraria a memória do computador. O `DataLoader` do PyTorch pega os milhares de pares (X, Y) gerados pela Sliding Window e os agrupa em blocos pequenos, chamados de `Batches`. A placa de vídeo processa um lote por vez, o que torna o treinamento computacionalmente viável.