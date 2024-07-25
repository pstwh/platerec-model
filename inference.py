import torch
from torchvision import transforms
from PIL import Image
import argparse
from config import decode
from model import EncoderDecoder

IMAGE_EMB_SIZE = 96
TEXT_EMB_SIZE = 64


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="License Plate Recognition Inference")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image"
    )
    args = parser.parse_args()

    model = EncoderDecoder()
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval().cuda()

    with torch.no_grad():
        image_tensor = preprocess_image(args.image_path).cuda()
        with torch.no_grad():
            output_indices, token_confidence = model.generate(image_tensor)
        predicted_text = decode(output_indices)

        print(f"Predicted license plate: {predicted_text}, {token_confidence}")


if __name__ == "__main__":
    main()
