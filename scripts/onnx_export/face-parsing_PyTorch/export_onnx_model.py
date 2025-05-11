import argparse
import torch
import warnings

from model import BiSeNet

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

parser = argparse.ArgumentParser(
    description="Export the SAM image encoder to an ONNX model."
)

parser.add_argument(
    "--checkpoint", type=str, required=True, help="The path to the SAM model checkpoint."
)

parser.add_argument(
    "--output", type=str, required=True, help="The filename to save the ONNX model to."
)

parser.add_argument(
    "--use-preprocess",
    action="store_true",
    help="Whether to preprocess the image by resizing, standardizing, etc.",
)

parser.add_argument(
    "--opset",
    type=int,
    default=17,
    help="The ONNX opset version to use. Must be >=11",
)


class OnnxModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        feat, feat16, feat32 = self.model(x)
        print(feat.shape)
        print(feat.argmax(1).shape)
        return feat.argmax(1)


def run_export(
    checkpoint: str,
    output: str,
    use_preprocess: bool,
    opset: int,
):
    print("Loading model...")
    model = BiSeNet(n_classes=19)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    onnx = OnnxModel(model)

    dummy_input = torch.randn(1, 3, 512, 512)

    _ = onnx(dummy_input)

    torch.onnx.export(
        onnx,
        dummy_input,
        output,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"Model exported to {output}")


if __name__ == "__main__":
    args = parser.parse_args()
    run_export(
        checkpoint=args.checkpoint,
        output=args.output,
        use_preprocess=args.use_preprocess,
        opset=args.opset,
    )
