
## platerec-model
platerec-model is a model for recognizing text from images, specifically designed for license plate recognition. The project utilizes a neural network architecture with an encoder-decoder setup and uses SAM (Sharpness-Aware Minimization) for optimizing the model training process. It's really lightweight using only a mobilenet v2 for encoder and a decoder transformer (gpt) for decoder. It is used in the platerec project.

The idea is to transform this training repository into a library that can be used easily.

### Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)
- [Model Architecture](#model-architecture)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/platerec-model.git
   cd platerec-model   
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Training

To train the model, use the following command:

```bash
python train.py --config_path config.yaml --model_checkpoint artifacts/trained_model.pth --device cuda --num_epochs 10
```

Dataset is expected to be in the following format:
```
├── 1.jpg
├── 1.txt
├── 2.jpg
├── 2.txt
├── 3.jpg
├── 3.txt
├── 4.jpg
└── 4.txt
```

- `--model_checkpoint`: Path to a pretrained model (.pth file) if you have.
- `--device`: The device to use for training (`cuda` or `cpu`). Defaults to `cuda` if available.
- `--num_epochs`: Number of epochs for training. Default is 10.

#### Inference

To perform inference with the trained model, use the following command:

```bash
python inference.py --model_path artifacts/trained_model.pth --tokenizer_path artifacts/tokenizer.json --image_path test_image.jpg
```

**Parameters:**
- `--model_path`: Path to the trained model checkpoint (.pth file).
- `--tokenizer_path`: Path to the tokenizer file (.json file).
- `--image_path`: Path to the image file for which text recognition is to be performed.

### Model Architecture

The platerec-model employs an encoder-decoder architecture with cross-attention mechanisms. The key components are:

- **Encoder:** Based on `mobilenet_v2` for feature extraction from images.
- **Decoder:** Utilizes an embedding layer, position encoding, and multiple decoder blocks with self-attention and cross-attention layers.
- **Loss Function:** Uses `cross_entropy` loss, with special handling for a specific index (`ignore_index of ~ token`).

