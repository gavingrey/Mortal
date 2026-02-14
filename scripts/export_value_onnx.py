#!/usr/bin/env python3
"""Export Brain+ValueNet to ONNX for placement prediction.

Usage:
    python scripts/export_value_onnx.py <brain.pth> <value.pth> <output.onnx> [--version 4] [--conv-channels 192] [--num-blocks 40]

Combines frozen Brain encoder with trained ValueNet head.
Input: obs (batch, 1012, 34) f32
Output: placement_logits (batch, 4) f32
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mortal'))
from model import Brain, ValueNet

try:
    from libriichi.consts import obs_shape
except ImportError:
    def obs_shape(version):
        return {1: (938, 34), 2: (942, 34), 3: (934, 34), 4: (1012, 34)}[version]


class ValueONNXWrapper(nn.Module):
    """Wrapper combining Brain+ValueNet for ONNX export."""

    def __init__(self, brain: Brain, value_net: ValueNet):
        super().__init__()
        self.brain = brain
        self.value_net = value_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        phi = self.brain(obs)            # (batch, 1024)
        return self.value_net(phi)       # (batch, 4)


def main():
    parser = argparse.ArgumentParser(description='Export Brain+ValueNet to ONNX')
    parser.add_argument('brain_path', help='Path to Brain checkpoint (.pth)')
    parser.add_argument('value_path', help='Path to ValueNet checkpoint (.pth)')
    parser.add_argument('output_path', help='Output ONNX file path')
    parser.add_argument('--version', type=int, default=4, help='Model version (default: 4)')
    parser.add_argument('--conv-channels', type=int, default=192, help='ResNet conv channels (default: 192)')
    parser.add_argument('--num-blocks', type=int, default=40, help='ResNet num blocks (default: 40)')
    parser.add_argument('--hidden-dim', type=int, default=None, help='ValueNet hidden dim override (auto-detected from checkpoint)')
    args = parser.parse_args()

    version = args.version
    channels, width = obs_shape(version)

    print(f"Model version: {version}")
    print(f"Obs shape: ({channels}, {width})")

    # Load Brain
    print(f"\nLoading Brain from {args.brain_path}")
    brain_checkpoint = torch.load(args.brain_path, map_location='cpu', weights_only=True)
    brain_sd = brain_checkpoint['mortal']  # Training checkpoint format

    brain = Brain(conv_channels=args.conv_channels, num_blocks=args.num_blocks, version=version)
    brain.load_state_dict(brain_sd)
    brain.eval()

    # Load ValueNet
    print(f"Loading ValueNet from {args.value_path}")
    value_checkpoint = torch.load(args.value_path, map_location='cpu', weights_only=True)
    value_sd = value_checkpoint['model']

    hidden_dim = args.hidden_dim or value_checkpoint.get('hidden_dim')
    print(f"  hidden_dim={'auto:' if args.hidden_dim is None else 'override:'}{hidden_dim}")
    value_net = ValueNet(hidden_dim=hidden_dim)
    value_net.load_state_dict(value_sd)
    value_net.eval()

    wrapper = ValueONNXWrapper(brain, value_net)
    wrapper.eval()

    # Validate wrapper
    print("\nValidating wrapper...")
    for batch_size in [1, 2, 4]:
        obs = torch.randn(batch_size, channels, width, dtype=torch.float32)

        with torch.no_grad():
            phi = brain(obs)
            ref_out = value_net(phi)
            wrap_out = wrapper(obs)

        diff = (ref_out - wrap_out).abs().max().item()
        print(f"  batch={batch_size}: max_diff={diff:.2e} {'OK' if diff < 1e-6 else 'WARNING'}")

    # Export to ONNX
    print(f"\nExporting to {args.output_path}")
    dummy_obs = torch.randn(2, channels, width, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (dummy_obs,),
        args.output_path,
        input_names=['obs'],
        output_names=['placement_logits'],
        dynamic_axes={
            'obs': {0: 'batch_size'},
            'placement_logits': {0: 'batch_size'},
        },
        opset_version=17,
    )

    # Verify with onnx
    print("Verifying ONNX model...")
    import onnx
    model = onnx.load(args.output_path)
    onnx.checker.check_model(model)
    print(f"  ONNX validated OK")

    # Verify with onnxruntime
    print("\nVerifying with onnxruntime...")
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(args.output_path)

    for batch_size in [1, 2, 4]:
        obs = torch.randn(batch_size, channels, width, dtype=torch.float32)

        with torch.no_grad():
            ref_out = wrapper(obs).numpy()

        onnx_out = session.run(None, {'obs': obs.numpy()})[0]
        diff = np.abs(ref_out - onnx_out).max()
        print(f"  batch={batch_size}: PyTorch vs ONNX max_diff={diff:.2e} {'OK' if diff < 1e-5 else 'WARNING'}")

    file_size = os.path.getsize(args.output_path)
    print(f"\nDone! ONNX saved to {args.output_path} ({file_size / 1024 / 1024:.1f} MB)")


if __name__ == '__main__':
    main()
