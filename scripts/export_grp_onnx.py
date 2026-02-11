#!/usr/bin/env python3
"""Export GRP model to ONNX format for use with tract runtime in Rust.

Usage:
    python scripts/export_grp_onnx.py <model.pth> <output.onnx>

The wrapper avoids PackedSequence by using raw GRU + fc directly,
accepting a simple (1, seq_len, 7) tensor as input.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

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
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

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

    # Validate at multiple sequence lengths
    print("Validating wrapper against original model...")
    for seq_len in [1, 3, 5, 8]:
        x = torch.randn(1, seq_len, 7, dtype=torch.float64)

        # Original model forward (uses pack_padded_sequence internally)
        with torch.no_grad():
            orig_out = grp([x.squeeze(0)])  # GRP.forward takes List[Tensor]

        # Wrapper forward
        with torch.no_grad():
            wrap_out = wrapper(x)

        diff = (orig_out - wrap_out).abs().max().item()
        print(f"  seq_len={seq_len}: max_diff={diff:.2e}", end="")
        if diff < 1e-10:
            print(" OK")
        else:
            print(f" WARNING: large difference!")

    # Export to ONNX
    print(f"\nExporting to {output_path}")
    dummy_input = torch.randn(1, 3, 7, dtype=torch.float64)
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

    # Verify the exported model
    print("Verifying exported ONNX model...")
    import onnxruntime as ort

    session = ort.InferenceSession(output_path)

    for seq_len in [1, 3, 5, 8]:
        x = torch.randn(1, seq_len, 7, dtype=torch.float64)

        # PyTorch reference
        with torch.no_grad():
            ref_out = wrapper(x).numpy()

        # ONNX inference
        onnx_out = session.run(None, {'input': x.numpy()})[0]

        diff = np.abs(ref_out - onnx_out).max()
        print(f"  seq_len={seq_len}: PyTorch vs ONNX max_diff={diff:.2e}", end="")
        if diff < 1e-6:
            print(" OK")
        else:
            print(f" WARNING: large difference!")

    print(f"\nDone! ONNX model saved to {output_path}")


if __name__ == '__main__':
    main()
