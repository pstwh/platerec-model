import string

tokens = "<" + string.ascii_uppercase + string.digits + " " + ">" + "~"
stoi = {ch: i for i, ch in enumerate(tokens)}
itos = {i: ch for i, ch in enumerate(tokens)}


def encode(s):
    return [stoi[ch] for ch in s]


def decode(idxs):
    return "".join([itos[idx] for idx in idxs])


block_size = 16
vocab_size = len(tokens)
