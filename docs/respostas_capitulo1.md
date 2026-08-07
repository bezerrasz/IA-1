# Respostas Teóricas - Capítulo 1

**1. O que caracteriza um Large Language Model (LLM)?**
Um LLM é um modelo de rede neural de grande escala, treinado em quantidades massivas de texto, projetado para entender, gerar e interagir com a linguagem humana. Sua principal característica é a capacidade de prever a próxima palavra (ou token) em uma sequência, capturando nuances complexas de gramática, semântica e contexto.

**2. Qual a diferença entre IA, Machine Learning e Deep Learning?**
* **IA (Inteligência Artificial):** O campo amplo de criar máquinas capazes de realizar tarefas que exigiriam inteligência humana.
* **Machine Learning (Aprendizado de Máquina):** Subcampo da IA focado em algoritmos que aprendem padrões a partir de dados, em vez de serem explicitamente programados.
* **Deep Learning (Aprendizado Profundo):** Subcampo do ML que utiliza Redes Neurais Artificiais com múltiplas camadas (profundas) para extrair padrões altamente complexos. Os LLMs são uma aplicação direta de Deep Learning.

**3. Como os modelos GPT estão organizados?**
A arquitetura GPT (Generative Pre-trained Transformer) baseia-se exclusivamente na parte do **Decoder** da arquitetura Transformer original. Eles são organizados em múltiplas camadas empilhadas de blocos de decodificação, utilizando mecanismos de *Autoatenção Mascarada* (Masked Self-Attention) para processar o texto de forma autorregressiva (da esquerda para a direita).

**4. Quais são as principais etapas do treinamento de um LLM?**
* **Pré-treinamento (Pre-training):** O modelo consome terabytes de texto cru para aprender a estrutura da linguagem e acumular conhecimento geral. O objetivo é a previsão do próximo token.
* **Ajuste Fino (Fine-tuning):** O modelo pré-treinado é atualizado com um conjunto menor e mais específico de dados para seguir instruções (Instruction fine-tuning) ou se especializar em uma tarefa (como classificação ou formatação).

**5. Qual é o fluxo geral de funcionamento de um modelo de linguagem?**
O texto de entrada (Prompt) é convertido em IDs numéricos (Tokenização) e, em seguida, transformado em vetores de contexto (Embeddings). Esses vetores passam pelas camadas de Atenção da rede neural, que calculam a relação entre as palavras. Por fim, o modelo gera uma distribuição de probabilidade para prever o próximo token. O token escolhido é adicionado à sequência e o ciclo se repete até atingir um limite de tamanho ou gerar um token de parada.