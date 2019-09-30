import string
from fastai.text import BaseTokenizer, UNK, BOS, Vocab, Tokenizer


class LetterTokenizer(BaseTokenizer):
    "Character level tokenizer function."
    def __init__(self, lang): pass
    def tokenizer(self, t):
        out = []
        i = 0
        while i < len(t):
            if t[i:].startswith(BOS):
                out.append(BOS)
                i += len(BOS)
            else:
                out.append(t[i])
                i += 1
        return out
            
    def add_special_cases(self, toks): pass


def get_tokenizer_with_vocab():
    itos = [UNK, BOS] + list(string.printable)
    vocab = Vocab(itos)
    tokenizer = Tokenizer(LetterTokenizer, pre_rules=[], post_rules=[])
    return tokenizer, vocab
