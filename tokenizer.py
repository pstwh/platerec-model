import json
import string


class Tokenizer:
    def __init__(self, special_tokens=None, block_size=32):
        self.base_tokens = list("<" + string.ascii_uppercase + string.digits + " >~")
        self.special_tokens = special_tokens or []
        self.tokens = self.base_tokens + self.special_tokens

        self._initialize_mappings()
        self.block_size = block_size
        self.vocab_size = len(self.tokens)
        self.ignore_index = self.stoi["~"]

    def _initialize_mappings(self):
        self.stoi = {ch: i for i, ch in enumerate(self.tokens)}
        self.itos = {i: ch for i, ch in enumerate(self.tokens)}

    @classmethod
    def from_config(cls, config):
        special_tokens = [f"[{c['name']}]" for c in config]
        return cls(special_tokens=special_tokens)

    def save_to_json(self, json_path):
        data = {
            "stoi": self.stoi,
            "itos": self.itos,
            "block_size": self.block_size,
            "vocab_size": self.vocab_size,
            "ignore_index": self.ignore_index,
            "special_tokens": self.special_tokens,
            "tokens": self.tokens,
        }

        try:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)
        except IOError as e:
            print(f"Error saving tokenizer to {json_path}: {e}")

    @classmethod
    def load_from_json(cls, json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading tokenizer from {json_path}: {e}")

        tokenizer = cls(
            special_tokens=data["special_tokens"], block_size=data["block_size"]
        )
        tokenizer.stoi = data["stoi"]
        tokenizer.itos = data["itos"]
        tokenizer.ignore_index = data["ignore_index"]
        tokenizer.tokens = data["tokens"]
        tokenizer.vocab_size = len(tokenizer.tokens)
        return tokenizer

    def add_special_token(self, token):
        if token in self.tokens:
            raise ValueError(f"Token {token} already exists.")

        self.tokens.append(token)
        self.special_tokens.append(token)
        self.stoi[token] = len(self.stoi)
        self.itos[len(self.itos)] = token
        self.vocab_size = len(self.tokens)

    def encode(self, s):
        i = 0
        encoded = []
        while i < len(s):
            match = None
            for token in self.special_tokens:
                if s[i : i + len(token)] == token:
                    match = token
                    break

            if match:
                encoded.append(self.stoi[match])
                i += len(match)
            elif s[i] in self.stoi:
                encoded.append(self.stoi[s[i]])
                i += 1
            else:
                raise ValueError(f"Unknown character or token: {s[i]}")

        return encoded

    def decode(self, indices):
        return "".join(self.itos[str(idx)] for idx in indices)


if __name__ == "__main__":
    import yaml

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    tokenizer = Tokenizer.from_config(config)
    tokenizer.save_to_json("artifacts/tokenizer.json")

    tokenizer = Tokenizer.load_from_json("artifacts/tokenizer.json")
    print(tokenizer.__dict__)
