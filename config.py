import string
import yaml

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)
    
tokens = list("<" + string.ascii_uppercase + string.digits + " " + ">" + "~")

special_tokens = [f"[{c['name']}]" for c in config]

tokens.extend(special_tokens)

stoi = {ch: i for i, ch in enumerate(tokens)}
itos = {i: ch for i, ch in enumerate(tokens)}

def encode(s):
    i = 0
    encoded = []
    while i < len(s):
        match = None
        for token in special_tokens:
            if s[i : i + len(token)] == token:
                match = token
                break
        
        if match:
            encoded.append(stoi[match])
            i += len(match)
        elif s[i] in stoi:
            encoded.append(stoi[s[i]])
            i += 1
        else:
            raise ValueError(f"Unknown character or token: {s[i]}")
    
    return encoded

def decode(idxs):
    return "".join([itos[idx] for idx in idxs])

block_size = 32
vocab_size = len(tokens)
ignore_index = tokens.index("~")

print('Ignore index:',ignore_index)