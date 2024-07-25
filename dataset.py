import torch
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
        text_path = str(self.data[idx])
        image_path = text_path.replace(".txt", ".jpg")
        with open(text_path) as f:
            text = f.read()
        text = ''.join(list(filter(lambda c: c in tokens, text)))
        text = text.replace('<', '').replace('>', '')
        text = "<" + text + ">"
        if len(text) < 17:
            text = text + (17 - len(text)) * "~"
        textf = text[1:]
        text = text[:-1]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image=np.array(image))["image"]

        text = encode(text)
        text = torch.Tensor(text).long()
        textf = encode(textf)
        textf = torch.Tensor(textf).long()

        return image, text, textf
