import argparse
from glob import glob
from random import shuffle

import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImageTextDataset
from model import EncoderDecoder
from sam import SAM
from tokenizer import Tokenizer
from transforms import transform, val_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Train platerec model.")

    parser.add_argument(
        "--config_path",
        default="config.yml",
        type=str,
        help="Path to the config file.",
    )

    parser.add_argument(
        "--model_checkpoint",
        type=str,
        help="A pretrained model filepath (.pth file)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use for training ('cuda' or 'cpu'). Defaults to 'cuda' if available, otherwise 'cpu'.",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs to train. Default is 10.",
    )

    return parser.parse_args()


def train(model, tokenizer, dataloader, optimizer, scheduler, device):
    model.train()
    losses = []
    progress_bar = tqdm(dataloader, desc="Training")

    for images, texts, texts_shifted in progress_bar:
        images, texts, texts_shifted = (
            images.to(device),
            texts.to(device),
            texts_shifted.to(device),
        )

        yp = model(texts, images)
        B, T, C = yp.shape
        yp = yp.view(B * T, C)
        y = texts_shifted.view(B * T)

        loss = F.cross_entropy(yp, y, ignore_index=tokenizer.ignore_index)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        yp = model(texts, images)
        B, T, C = yp.shape
        yp = yp.view(B * T, C)
        y = texts_shifted.view(B * T)

        loss = F.cross_entropy(yp, y, ignore_index=tokenizer.ignore_index)
        loss.backward()
        optimizer.second_step(zero_grad=True)
        scheduler.step()

        losses.append(loss.item())
        mean_loss = torch.tensor(losses).mean().item()
        progress_bar.set_description(f"Loss: {mean_loss:.4f}")

    return mean_loss


def validate(model, tokenizer, val_dataloader, device):
    model.eval()
    val_losses = []
    total_tokens = 0
    correct_tokens = 0
    total_words = 0
    correct_words = 0

    with torch.no_grad():
        for images, texts, textsf in val_dataloader:
            images, texts, textsf = (
                images.to(device),
                texts.to(device),
                textsf.to(device),
            )

            yp = model(texts, images)
            B, T, C = yp.shape
            yp = yp.view(B * T, C)
            y = textsf.view(B * T)

            val_loss = F.cross_entropy(yp, y, ignore_index=tokenizer.ignore_index)
            val_losses.append(val_loss.item())

            predictions = torch.argmax(yp, dim=1)
            mask = y != tokenizer.ignore_index
            correct_tokens += (predictions[mask] == y[mask]).sum().item()
            total_tokens += mask.sum().item()

            predictions = predictions.view(B, T)
            y = y.view(B, T)

            word_correct = 0
            for i in range(B):
                valid_indices = y[i] != tokenizer.ignore_index

                valid_predictions = predictions[i][valid_indices]
                valid_targets = y[i][valid_indices]

                if torch.equal(valid_predictions, valid_targets):
                    word_correct += 1

            correct_words += word_correct
            total_words += B

    avg_val_loss = torch.tensor(val_losses).mean().item()
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    word_accuracy = correct_words / total_words if total_words > 0 else 0

    return avg_val_loss, token_accuracy, word_accuracy


def main():
    args = parse_args()

    device = args.device
    num_epochs = args.num_epochs

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    tokenizer = Tokenizer.from_config(config)
    print("Ignore index: ", tokenizer.ignore_index)

    data = []
    for c in config:
        for p in c["paths"]:
            files = glob(p)
            files = list(map(lambda f: (c["name"], f), files))
            data.extend(files)
    print(f"Total data: {len(data)}")
    shuffle(data)

    dataset_size = len(data)
    dataset = ImageTextDataset(
        data[: int(dataset_size * 0.95)], transform=transform, tokenizer=tokenizer
    )
    val_dataset = ImageTextDataset(
        data[int(dataset_size * 0.95) :], transform=val_transform, tokenizer=tokenizer
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    model = EncoderDecoder(tokenizer=tokenizer)
    if args.model_checkpoint:
        model.load_state_dict(torch.load(args.model_checkpoint))
    model = model.to(device)
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=1e-4)
    total_steps = num_epochs * len(dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        train_loss = train(model, tokenizer, dataloader, optimizer, scheduler, device)
        val_loss, token_acc, word_acc = validate(
            model, tokenizer, val_dataloader, device
        )

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Token Accuracy: {token_acc:.4%}")
        print(f"Word Accuracy: {word_acc:.4%}")

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), "best.pth")
            best_val_loss = val_loss
            print(f"Best Val Loss Now: {val_loss:.4f}!")

        torch.save(model.state_dict(), "last.pth")


if __name__ == "__main__":
    main()
