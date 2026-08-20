import sys
import os

# Resolve o caminho para achar a pasta src
caminho_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(caminho_src)

import torch
from tokeniz import raw_text, vocab, SimpleTokenizerV2, create_dataloader_v1

# INICIALIZAÇÃO DO TOKENIZER 
tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

#TESTE DO TOKENIZER
print(text)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))

# CRIANDO O DATALOADER 
max_length = 4 
dataloader = create_dataloader_v1(raw_text, tokenizer, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

#TESTE DO DATALOADER
print(inputs)
print(targets)

# DEFINIDO O TAMANHO DO VOCABULÁRIO 
vocab_size = len(vocab)
embedding_dim = 256     
context_length = max_length

# CRIANDO AS CAMADAS 
token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, embedding_dim)

# CRIANDO OS EMBEDDINGS
tok_embeddings = token_embedding_layer(inputs)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
input_embeddings = tok_embeddings + pos_embeddings

#TESTE DO EMBEDDING
print(input_embeddings.shape)