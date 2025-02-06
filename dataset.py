import torch
import re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from config import encode, tokens


class ImageTextDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cls, image_path = self.data[idx]
        text_path = text_path = re.sub(r"\.(jpg|jpeg|png)$", ".txt", image_path, flags=re.IGNORECASE)

        with open(text_path, encoding="utf-8") as f:
            text = f.read()
            text = text.replace("<", "")
            text = text.replace(">", "")

        tokenized_text = []
        i = 0
        while i < len(text):
            match = None
            for token in tokens:
                if text[i : i + len(token)] == token:
                    match = token
                    break

            if match:
                tokenized_text.append(match)
                i += len(match)
            elif text[i] in tokens:
                tokenized_text.append(text[i])
                i += 1
            else:
                i += 1

        tokenized_text = ["<", f"[{cls}]"] + tokenized_text + [">"]
        if len(tokenized_text) < 33:
            tokenized_text += ["~"] * (33 - len(tokenized_text))

        text_shifted = tokenized_text[1:]
        text = tokenized_text[:-1]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image=np.array(image))["image"]

        text = encode(text)
        text = torch.tensor(text, dtype=torch.long)
        text_shifted = encode(text_shifted)
        text_shifted = torch.tensor(text_shifted, dtype=torch.long)

        return image, text, text_shifted
