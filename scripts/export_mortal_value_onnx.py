#!/usr/bin/env python3
"""Export Mortal Brain + DQN value head to ONNX for use with tract runtime in Rust.

Usage:
    python scripts/export_mortal_value_onnx.py <mortal.pth> <output.onnx>

Exports a single ONNX model that maps game observation → scalar state value.
Input:  (batch, obs_channels, 34) float32
Output: (batch, 1) float32 — scalar state value
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Add mortal/ to path so we can import model.py directly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mortal'))
from model import Brain, DQN


class MortalValueExportWrapper(nn.Module):
    """Wrapper that extracts only the value head output from Brain + DQN."""

    def __init__(self, brain, dqn):
        super().__init__()
        self.brain = brain
        self.dqn = dqn

    def forward(self, obs):                       # (batch, C, 34)
        phi = self.brain(obs)                      # (batch, 1024) for v2/3/4
        if self.dqn.version == 4:
            out = self.dqn.net(phi)                # (batch, 1 + ACTION_SPACE)
            return out[:, :1]                      # (batch, 1) — value only
        else:
            return self.dqn.v_head(phi)            # (batch, 1)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <mortal.pth> <output.onnx>")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading Mortal checkpoint from {model_path}")
    state = torch.load(model_path, map_location='cpu', weights_only=True)

    # Extract config and version
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    conv_channels = cfg['resnet']['conv_channels']
    num_blocks = cfg['resnet']['num_blocks']
    print(f"Detected: version={version}, conv_channels={conv_channels}, num_blocks={num_blocks}")

    # Get obs shape for this version
    from libriichi.consts import obs_shape
    obs_channels, obs_width = obs_shape(version)
    print(f"Obs shape: ({obs_channels}, {obs_width})")

    # Build and load models
    brain = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks)
    dqn = DQN(version=version)
    brain.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    brain.eval()
    brain.freeze_bn(True)
    dqn.eval()

    # Create wrapper
    wrapper = MortalValueExportWrapper(brain, dqn)
    wrapper.eval()

    # Validate wrapper matches direct computation
    print("\nValidating wrapper against direct Brain → v_head computation...")
    for batch_size in [1, 2, 4]:
        x = torch.randn(batch_size, obs_channels, obs_width)
        with torch.no_grad():
            # Direct computation
            phi = brain(x)
            if dqn.version == 4:
                out = dqn.net(phi)
                direct_value = out[:, :1]
            else:
                direct_value = dqn.v_head(phi)

            # Wrapper computation
            wrapper_value = wrapper(x)

        diff = (direct_value - wrapper_value).abs().max().item()
        print(f"  batch={batch_size}: max_diff={diff:.2e}", end="")
        if diff < 1e-10:
            print(" OK")
        else:
            print(f" WARNING: large difference!")

    # Print output range on sample inputs for scale verification
    print("\nOutput range check (for scale compatibility with placement points [6,4,2,0]):")
    for batch_size in [1, 8, 32]:
        x = torch.randn(batch_size, obs_channels, obs_width)
        with torch.no_grad():
            values = wrapper(x)
        print(f"  batch={batch_size}: min={values.min().item():.4f}, max={values.max().item():.4f}, "
              f"mean={values.mean().item():.4f}, std={values.std().item():.4f}")

    # Export to ONNX as float32
    print(f"\nExporting to {output_path} (float32)")
    dummy_input = torch.randn(2, obs_channels, obs_width, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=['obs'],
        output_names=['value'],
        dynamic_axes={
            'obs': {0: 'batch_size'},
            'value': {0: 'batch_size'},
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
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, obs_channels, obs_width, dtype=torch.float32)
        with torch.no_grad():
            ref_out = wrapper(x).numpy()
        onnx_out = session.run(None, {'obs': x.numpy()})[0]
        diff = np.abs(ref_out - onnx_out).max()
        print(f"  batch={batch_size}: PyTorch vs ONNX max_diff={diff:.2e}", end="")
        if diff < 1e-4:
            print(" OK")
        else:
            print(f" WARNING: large difference!")

    file_size = os.path.getsize(output_path)
    print(f"\nDone! ONNX model saved to {output_path} ({file_size / 1024 / 1024:.1f} MB)")


if __name__ == '__main__':
    main()
