try:
    from .config import CORPUS_PATH, SPECIAL_TOKENS
    from .dataloader import create_dataloader_v1
    from .dataset import GPTDatasetV1
    from .tokenizer import SimpleTokenizerV2, build_vocab, tokenize_text
except ImportError:  # Permite executar o arquivo diretamente a partir de src/.
    from config import CORPUS_PATH, SPECIAL_TOKENS
    from dataloader import create_dataloader_v1
    from dataset import GPTDatasetV1
    from tokenizer import SimpleTokenizerV2, build_vocab, tokenize_text

# 1. Ler o texto e criar o vocabulário.
with CORPUS_PATH.open("r", encoding="utf-8") as f:
    raw_text = f.read()

# Separa palavras, pontuação e tokens especiais para extrair os tokens únicos.
preprocessed_text = tokenize_text(raw_text, SPECIAL_TOKENS)

# Cria uma lista ordenada sem palavras repetidas e gera o dicionário
vocab = build_vocab(preprocessed_text, SPECIAL_TOKENS)
all_tokens = list(vocab)
