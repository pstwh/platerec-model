import argparse
from pathlib import Path
from random import shuffle

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImageTextDataset
from model import EncoderDecoder
from sam import SAM
from transforms import transform, val_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Train platerec model.")

    parser.add_argument(
        "--dataset_paths",
        type=str,
        nargs="+",
        required=True,
        help="A list of directories to specify input data files.",
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


def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    losses = []
    progress_bar = tqdm(dataloader, desc="Training")

    for images, texts, textsf in progress_bar:
        images, texts, textsf = images.to(device), texts.to(device), textsf.to(device)

        yp = model(texts, images)
        B, T, C = yp.shape
        yp = yp.view(B * T, C)
        y = textsf.view(B * T)

        loss = F.cross_entropy(yp, y, ignore_index=39)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        yp = model(texts, images)
        B, T, C = yp.shape
        yp = yp.view(B * T, C)
        y = textsf.view(B * T)

        loss = F.cross_entropy(yp, y, ignore_index=39)
        loss.backward()
        optimizer.second_step(zero_grad=True)
        scheduler.step()

        losses.append(loss.item())
        mean_loss = torch.tensor(losses).mean().item()
        progress_bar.set_description(f"Loss: {mean_loss:.4f}")

    return mean_loss


def validate(model, val_dataloader, device):
    model.eval()
    val_losses = []
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

            val_loss = F.cross_entropy(yp, y, ignore_index=39)
            val_losses.append(val_loss.item())

    return torch.tensor(val_losses).mean().item()


def main():
    args = parse_args()

    device = args.device
    num_epochs = args.num_epochs

    data = list(map(lambda d: list(Path(d).glob("*.txt")), args.dataset_paths))
    data = sum(data, [])
    print(f"Total data: {len(data)}")
    shuffle(data)

    dataset_size = len(data)
    dataset = ImageTextDataset(data[: int(dataset_size * 0.95)], transform=transform)
    val_dataset = ImageTextDataset(
        data[int(dataset_size * 0.95) :], transform=val_transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    model = EncoderDecoder()
    if args.model_checkpoint:
        model.load_state_dict(torch.load(args.model_checkpoint))
    model = model.to(device)
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=1e-4)
    total_steps = num_epochs * len(dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        train_loss = train(model, dataloader, optimizer, scheduler, device)
        val_loss = validate(model, val_dataloader, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), "best.pth")
            best_val_loss = val_loss
            print(f"Best Val Loss Now: {val_loss:.4f}!")

        torch.save(model.state_dict(), "last.pth")


if __name__ == "__main__":
    main()
