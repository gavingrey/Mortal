#!/usr/bin/env python3
"""Export GRP model to ONNX format for use with tract runtime in Rust.

Usage:
    python scripts/export_grp_onnx.py <model.pth> <output.onnx>

The wrapper avoids PackedSequence by using raw GRU + fc directly,
accepting a simple (1, seq_len, 7) tensor as input.
Exports as float32 for broad ONNX runtime compatibility (tract, onnxruntime).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Add mortal/ to path so we can import model.py directly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mortal'))
from model import GRP


class GRPExportWrapper(nn.Module):
    """Wrapper for ONNX export that avoids PackedSequence."""

    def __init__(self, grp: GRP):
        super().__init__()
        self.rnn = grp.rnn
        self.fc = grp.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, seq_len, 7)
        _, state = self.rnn(x)  # state: (num_layers, 1, hidden)
        state = state.transpose(0, 1).flatten(1)  # (1, num_layers*hidden)
        return self.fc(state)  # (1, 24)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <model.pth> <output.onnx>")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading GRP model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Handle full checkpoint (with 'model', 'optimizer', etc.) or bare state dict
    if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        print("Detected full checkpoint format, extracting 'model' key")
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Detect hidden_size and num_layers from state dict
    # GRU weight_ih_l0 shape: (3*hidden_size, input_size)
    weight_key = 'rnn.weight_ih_l0'
    if weight_key not in state_dict:
        print(f"Error: expected key '{weight_key}' in state dict")
        print(f"Available keys: {list(state_dict.keys())}")
        sys.exit(1)

    hidden_size = state_dict[weight_key].shape[0] // 3
    num_layers = sum(1 for k in state_dict if k.startswith('rnn.weight_ih_l'))
    print(f"Detected: hidden_size={hidden_size}, num_layers={num_layers}")

    grp = GRP(hidden_size=hidden_size, num_layers=num_layers)
    grp.load_state_dict(state_dict)
    grp.eval()

    wrapper = GRPExportWrapper(grp)
    wrapper.eval()

    # Validate wrapper matches original in f64
    print("Validating wrapper against original model (f64)...")
    for seq_len in [1, 3, 5, 8]:
        x = torch.randn(1, seq_len, 7, dtype=torch.float64)
        with torch.no_grad():
            orig_out = grp([x.squeeze(0)])
            wrap_out = wrapper(x)
        diff = (orig_out - wrap_out).abs().max().item()
        print(f"  seq_len={seq_len}: max_diff={diff:.2e}", end="")
        if diff < 1e-10:
            print(" OK")
        else:
            print(f" WARNING: large difference!")

    # Save f64 reference outputs before converting to f32
    print("\nComputing f64 reference outputs for precision comparison...")
    ref_inputs = [torch.randn(1, seq_len, 7, dtype=torch.float64) for seq_len in [1, 3, 5, 8]]
    with torch.no_grad():
        ref_outputs = [wrapper(x) for x in ref_inputs]

    # Convert to float32 for ONNX export (tract and onnxruntime support f32 GRU)
    print("Converting model to float32 for export...")
    wrapper.float()  # in-place conversion

    # Validate f32 vs f64 precision loss is acceptable
    print("Validating f32 precision loss...")
    for x64, out64, seq_len in zip(ref_inputs, ref_outputs, [1, 3, 5, 8]):
        x32 = x64.float()
        with torch.no_grad():
            out32 = wrapper(x32)
        diff = (out64.float() - out32).abs().max().item()
        print(f"  seq_len={seq_len}: f64 vs f32 max_diff={diff:.2e}", end="")
        if diff < 1e-3:
            print(" OK")
        else:
            print(f" WARNING: large difference!")

    # Export to ONNX as float32
    print(f"\nExporting to {output_path} (float32)")
    dummy_input = torch.randn(1, 3, 7, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={
            'input': {1: 'seq_len'},
            'logits': {},
        },
        opset_version=17,
    )

    # Verify with onnx checker
    print("Verifying exported ONNX model structure...")
    import onnx
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    print(f"  ONNX model validated OK")
    print(f"  Inputs: {[i.name for i in model.graph.input]}")
    print(f"  Outputs: {[o.name for o in model.graph.output]}")

    # Verify with onnxruntime
    print("Verifying with onnxruntime...")
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    for seq_len in [1, 3, 5, 8]:
        x = torch.randn(1, seq_len, 7, dtype=torch.float32)
        with torch.no_grad():
            ref_out = wrapper(x).numpy()
        onnx_out = session.run(None, {'input': x.numpy()})[0]
        diff = np.abs(ref_out - onnx_out).max()
        print(f"  seq_len={seq_len}: PyTorch vs ONNX max_diff={diff:.2e}", end="")
        if diff < 1e-5:
            print(" OK")
        else:
            print(f" WARNING: large difference!")

    file_size = os.path.getsize(output_path)
    print(f"\nDone! ONNX model saved to {output_path} ({file_size / 1024:.1f} KB)")


if __name__ == '__main__':
    main()
