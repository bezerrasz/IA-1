# Avaliação de respostas do modelo em português

Checkpoint: `gpt_portugues_v2.pt`; época 29.

SHA-256 do conjunto de casos: `1973f3bad2e60b8d9be223726b9fdacbe70ee12ac6ad4b7ff0d721654bfa990e`.

Geração greedy livre, sem fornecer a resposta esperada ao modelo.
A comparação normaliza caixa e separa pontuação e operadores.

- Regressão (perguntas vistas no treino): 8/8;
- Teste controlado: 87/228 (38.16% de correspondência exata);

- Somas do teste: 78/100;
- Demais perguntas do teste: 9/128;

A pontuação agregada é influenciada pelos exemplos de soma. A taxa nas
demais perguntas é registrada separadamente para não ocultar dificuldades
com variações de linguagem.

## Perguntas observadas no terminal

Estas entradas estão no treino e verificam reprodução dos exemplos.

| Pergunta | Resposta gerada | Igual à referência |
|---|---|---|
| Oi | olá! tudo bem. como posso ajudar? | Sim |
| O que você pode fazer? | posso ajudar com explicações simples sobre estudos e o projeto. | Sim |
| ok | certo. pode perguntar. | Sim |
| Quanto é 1+1? | 2. | Sim |
| cara, fala direto | certo. vou responder de forma curta. | Sim |
| Como está? | estou pronto para ajudar. como posso ajudar você? | Sim |
| olá, tudo bem? | olá! tudo bem. como posso ajudar? | Sim |
| o que é um token? | um token é uma unidade de texto, como uma palavra ou um sinal. | Sim |

## Limites da avaliação

O teste contém perguntas diferentes das de treino e validação após
tokenização, mas compartilha intenções, perguntas-base e respostas.
Várias entradas são versões com prefixos diferentes de uma pergunta-base.
Portanto, o resultado mede variações controladas desse corpus sintético,
não capacidade de conversa geral nem generalização para assuntos novos.

As somas de números de 0 a 9 foram incluídas como exemplos supervisionados.
A inferência é feita apenas pelo GPT; acertar essas somas não demonstra
um algoritmo de cálculo, e contas fora desse domínio podem falhar.

Correspondência exata pode rejeitar uma paráfrase correta. Por outro lado,
memorizar respostas pode produzir boa pontuação sem compreensão ampla.

## Exemplos do teste controlado

| Pergunta | Referência | Resposta gerada | Igual |
|---|---|---|---|
| por favor, responda: o que é uma função de perda? | a função de perda mede a diferença entre a previsão do modelo e o resultado esperado. o treinamento tenta reduzir essa diferença. | treinar ajusta os pesos usando exemplos gerados e transforme informações. | Não |
| por favor, responda: quanto é 6+1? | 7. | 7. | Sim |
| por favor, responda: por que não usar só os resultados de treino? | resultados de treino mostram o ajuste aos exemplos apresentados. exemplos separados ajudam a verificar se o comportamento se mantém em outros textos. | resultados e evita atribuir ao modelo. | Não |
| por favor, responda: quanto é 5+3? | 8. | 8. | Sim |
| por favor, responda: quanto é 2+3? | 5. | 5. | Sim |
| por favor, responda: quanto é 3+2? | 5. | 5. | Sim |
| por favor, responda: quanto é 1+1? | 2. | 2. | Sim |
| por favor, responda: quanto é 4+0? | 4. | 4. | Sim |
| por favor, responda: quanto é 3+9? | 12. | 12. | Sim |
| por favor, responda: o que é um tensor? | um tensor organiza valores em dimensões. uma sequência de lotes e vetores pode ser representada por um tensor. | o modelo de linguagem aprende padrões em textos para prever o próximo token de uma sequência. | Não |
| por favor, responda: quanto é 0+4? | 4. | 0 de entrada. | Não |
| por favor, responda: como guardar o resultado do projeto? | guarde o código, a documentação e os relatórios no repositório. mantenha os pesos treinados em arquivos identificados. | o projeto. | Não |
| por favor, responda: cara, fala direto | certo. vou responder de forma curta. | sou um modelo pequeno e posso errar. | Não |
| por favor, responda: quanto é 6+5? | 11. | 11. | Sim |
| por favor, responda: quanto é 9+0? | 9. | 9. | Sim |
| por favor, responda: por que fazer testes? | os testes verificam se o programa apresenta o comportamento esperado. eles ajudam a encontrar erros após mudanças no código. | o programa salva o melhor checkpoint e usa os pesos aprendidos para produzir novas sequências de texto. | Não |
| por favor, responda: quanto é 3+4? | 7. | 7. | Sim |
| por favor, responda: como está? | estou pronto para ajudar. como posso ajudar você? | estou pronto para ajudar. como posso ajudar você? | Sim |
| por favor, responda: escreva uma frase sobre colaboração | um grupo aprende melhor quando compartilha dúvidas, testa suas ideias e registra as decisões do projeto. | estudar um marcador de fim ou um computador execute a ordem das amostras de treinamento. | Não |
| qual é a função da atenção? | a atenção combina informações dos tokens do contexto. | a normalização ajuda a normalização ajuda a controlar a escala das ativações. isso contribui para a estabilidade do processamento e do treinamento. | Não |
| posso confiar em todas as suas respostas? | sou um modelo pequeno e posso errar. confira as respostas. | um token é uma unidade de texto, como uma palavra ou um sinal. | Não |
| por favor, responda: o que é seleção greedy? | a seleção greedy escolhe o token com maior escore a cada passo. ela pode produzir respostas repetitivas. | a cada dia ajuda a organizar o conhecimento e registra. | Não |
| por favor, responda: tchau | até logo! bons estudos. | até logo! bons estudos. | Sim |
| por favor, responda: o que é um modelo de linguagem? | um modelo de linguagem aprende padrões em textos para prever o próximo token de uma sequência. | treinar ajusta demais aos exemplos para prever um tensor. | Não |
| por favor, responda: o que acontece quando eu escrevo uma pergunta? | o programa transforma a pergunta em tokens, executa o modelo e gera a resposta usando os pesos salvos. | o programa, executa o programa seleciona um modelo e o limite ou gerar o melhor projeto. | Não |
| por favor, responda: para que serve o tokenizador? | o tokenizador divide o texto em unidades e transforma essas unidades em identificadores numéricos. | o modelo recebe mais tokens para a previsão e precisa realizar mais operações. a atenção pode aumentar bastante o uso de memória. | Não |
| por favor, responda: como faço para usar os pesos salvos? | carregue o checkpoint correspondente ao idioma e execute a geração com o vocabulário salvo junto com o modelo. | carregue o erro. confira as previsões dos testes ajudam a verificar se o experimento não demonstrou. | Não |
| por favor, responda: você pode ajudar nos estudos de inteligência artificial? | posso tentar produzir explicações simples sobre o projeto. confira as respostas com os materiais de estudo e os testes. | posso ajudar você? | Não |
| por favor, responda: quanto é 4+3? | 7. | 7. | Sim |
| por favor, responda: quanto é 6+3? | 9. | 9. | Sim |
