from collections import Counter


class AttributeVocabulary:
    """Minimal vocabulary API used by SequencePAR without torchtext."""

    def __init__(self, counter, specials=("<bos>", "<eos>", "<pad>"), min_freq=1):
        self._itos = list(specials)
        self._itos.extend(
            token for token, count in counter.items()
            if count >= min_freq and token not in specials
        )
        self._stoi = {token: index for index, token in enumerate(self._itos)}

    def __getitem__(self, token):
        return self._stoi[token]

    def __len__(self):
        return len(self._itos)

    def lookup_token(self, index):
        return self._itos[index]


def attribute_vocab(words, specials=("<bos>", "<eos>", "<pad>")):
    return AttributeVocabulary(Counter(words), specials=specials, min_freq=1)
