import argparse

import onnx
import torch
from onnxconverter_common import float16

from model import EncoderDecoder
from tokenizer import Tokenizer


def export_onnx(model_path, tokenizer_path, encoder_output, decoder_output):
    tokenizer = Tokenizer.load_from_json(tokenizer_path)
    model = EncoderDecoder(tokenizer=tokenizer)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.eval().cuda()

    encoder_input = torch.randn(1, 3, 224, 224).cuda()
    text_input = torch.randint(0, tokenizer.vocab_size, (1, 32)).cuda()
    decoder_input = model.encoder(encoder_input)

    torch.onnx.export(
        model.encoder,
        encoder_input,
        encoder_output,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    torch.onnx.export(
        model.decoder,
        (text_input, decoder_input),
        decoder_output,
        export_params=True,
        opset_version=15,
        do_constant_folding=False,
        input_names=["input", "context"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "seq"},
            "context": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )


def convert_to_fp16(encoder_output, decoder_output):
    model = onnx.load(encoder_output)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, encoder_output.replace(".onnx", "_fp16.onnx"))

    model = onnx.load(decoder_output)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, decoder_output.replace(".onnx", "_fp16.onnx"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--encoder_output", type=str, default="encoder.onnx")
    parser.add_argument("--decoder_output", type=str, default="decoder.onnx")
    args = parser.parse_args()

    export_onnx(
        args.model_path, args.tokenizer_path, args.encoder_output, args.decoder_output
    )
    convert_to_fp16(args.encoder_output, args.decoder_output)
