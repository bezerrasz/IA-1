import re

# 1. LER O TEXTO E CRIAR O VOCABULÁRIO (O passo que estava faltando)
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Separa todas as palavras e pontuações para extrair os tokens únicos
preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]

# Cria uma lista ordenada sem palavras repetidas e gera o dicionário
all_tokens = sorted(list(set(preprocessed_text)))
vocab = {token: integer for integer, token in enumerate(all_tokens)}

# CRIAÇÃO DA CLASSE SIMPLETOKENIZERV1
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# TOCANIZANDO UM TEXTO DE EXEMPLO
tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

# DECODE O TEXTO DE VOLTA
tokenizer.decode(ids)
print(tokenizer.decode(ids))

#ERRO: KeyError: 'It's' - Isso acontece porque o token "It's" não está no vocabulário.
tokenizer = SimpleTokenizerV1(vocab)

text = "Hello, do you like tea. Is this-- a test?"

tokenizer.encode(text)

