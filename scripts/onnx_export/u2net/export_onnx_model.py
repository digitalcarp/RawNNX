import argparse
import torch.nn as nn
import torch.onnx

from model import U2NET
from model import U2NETP

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False


# Wrapper model to optimize and cleanup outputs
class OnnxModel(nn.Module):
    def __init__(self, model: U2NET | U2NETP):
        super().__init__()
        self.model = model

    # Disable computation of gradients to optimize inference
    @torch.no_grad()
    def forward(self, input_image: torch.Tensor):
        # Cleanup the output to only the mask
        d1, d2, d3, d4, d5, d6, d7 = self.model(input_image)
        # Output: Predicted SOD probability map
        # Needs to be normalised => (d- min(d)) / (max(d) - min(d))
        return d1[:, 0, :, :]


def run_export(
    model_type: str,
    checkpoint: str,
    output: str,
    opset: int,
    gelu_approximate: bool,
):
    if model_type == 'u2net':
        print("Loading U2NET model...")
        net = U2NET(3,1)
    elif model_type == 'u2netp':
        print("Loading U2NETP model...")
        net = U2NETP(3,1)

    if gelu_approximate:
        for n, m in net.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    net.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    net.eval()

    onnx_model = OnnxModel(net) 

    batch_size = 1
    x = torch.randn(batch_size, 3, 320, 320, dtype=torch.float)
    _ = onnx_model(x)

    input_names = ['input']
    output_names = ['output']
    dynamic_axes = {'input' : {0: 'batch_size'}, 'output': {0: 'batch_size'}}

    torch.onnx.export(
        onnx_model, x, output,
        export_params=True, opset_version=opset, do_constant_folding=True,
        input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)

    if onnxruntime_exists:
        ort_inputs = {'input': x.detach().numpy()}
        providers = ["CPUExecutionProvider"]

        ort_session = onnxruntime.InferenceSession(output, providers=providers)
        _ = ort_session.run(None, ort_inputs)

        print("Model has successfully been run with ONNXRuntime.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export U2-Net PyTorch to ONNX")

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the U2-Net model checkpoint"
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Filename to save the ONNX model to"
    )

    parser.add_argument(
        "--model-type", type=str, required=True,
        help="In ['u2net', 'u2netp']. Which type of U2-Net model to export.",
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version to use. Must be >=11",
    )

    parser.add_argument(
        "--quantize-out",
        type=str,
        default=None,
        help=(
            "If set, will quantize the model and save it with this name. "
            "Quantization is performed with quantize_dynamic from onnxruntime.quantization.quantize."
        ),
    )

    parser.add_argument(
        "--gelu-approximate",
        action="store_true",
        help=(
            "Replace GELU operations with approximations using tanh. Useful "
            "for some runtimes that have slow or unimplemented erf ops, used in GELU."
        ),
    )
    
    args = parser.parse_args()
    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output=args.output,
        opset=args.opset,
        gelu_approximate=args.gelu_approximate)

    if args.quantize_out is not None:
        assert onnxruntime_exists, "onnxruntime is required to quantize the model."
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

        print(f"Quantizing model and writing to {args.quantize_out}...")
        quantize_dynamic(
            model_input=args.output,
            model_output=args.quantize_out,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        print("Done!")
