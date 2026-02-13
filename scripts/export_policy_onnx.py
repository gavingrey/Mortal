#!/usr/bin/env python3
"""Export Brain+DQN policy model to ONNX format for use with tract runtime in Rust.

Usage:
    python scripts/export_policy_onnx.py <model.pth> <output.onnx> [--version 4] [--conv-channels 192] [--num-blocks 40]

The wrapper combines Brain (ResNet encoder) and DQN (dueling Q-network) into a
single ONNX model that maps (obs, mask) -> q_values.

Exports as float32 for ONNX runtime compatibility (tract, onnxruntime).
Mask input is float32 (0.0/1.0) and converted to bool internally, since ONNX
doesn't handle bool inputs well across all runtimes.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Add mortal/ to path so we can import model.py directly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mortal'))
from model import Brain, DQN

# Import obs_shape from libriichi if available, otherwise define locally
try:
    from libriichi.consts import obs_shape
except ImportError:
    def obs_shape(version):
        return {1: (938, 34), 2: (942, 34), 3: (934, 34), 4: (1012, 34)}[version]


class PolicyONNXWrapper(nn.Module):
    """Wrapper combining Brain+DQN for ONNX export.

    Accepts obs as float32 (batch, channels, 34) and mask as float32 (batch, 46)
    where mask values are 0.0 or 1.0 (converted to bool internally).
    Returns q_values as float32 (batch, 46).
    """

    def __init__(self, brain: Brain, dqn: DQN):
        super().__init__()
        self.brain = brain
        self.dqn = dqn

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        phi = self.brain(obs)           # (batch, 1024)
        mask_bool = mask.to(torch.bool) # float32 -> bool for DQN
        return self.dqn(phi, mask_bool) # (batch, 46)


def main():
    parser = argparse.ArgumentParser(description='Export Brain+DQN policy model to ONNX')
    parser.add_argument('model_path', help='Path to model checkpoint (.pth)')
    parser.add_argument('output_path', help='Output ONNX file path')
    parser.add_argument('--version', type=int, default=4, help='Model version (1-4, default: 4)')
    parser.add_argument('--conv-channels', type=int, default=192, help='ResNet conv channels (default: 192)')
    parser.add_argument('--num-blocks', type=int, default=40, help='ResNet num blocks (default: 40)')
    args = parser.parse_args()

    if args.version not in [1, 2, 3, 4]:
        parser.error(f"--version must be 1-4, got {args.version}")

    version = args.version
    channels, width = obs_shape(version)

    print(f"Model version: {version}")
    print(f"Obs shape: ({channels}, {width})")
    print(f"Conv channels: {args.conv_channels}, Num blocks: {args.num_blocks}")

    # Load checkpoint
    print(f"\nLoading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=True)

    # Training checkpoints use 'mortal' (Brain) and 'current_dqn' (DQN) keys.
    # Exported model dicts may use 'brain.' and 'dqn.' prefixed keys under 'model'.
    if 'mortal' in checkpoint and 'current_dqn' in checkpoint:
        print("Detected training checkpoint format ('mortal' + 'current_dqn' keys)")
        brain_sd = checkpoint['mortal']
        dqn_sd = checkpoint['current_dqn']
    elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        print("Detected exported model format ('model' key with brain./dqn. prefixes)")
        state_dict = checkpoint['model']
        brain_sd = {k.removeprefix('brain.'): v for k, v in state_dict.items() if k.startswith('brain.')}
        dqn_sd = {k.removeprefix('dqn.'): v for k, v in state_dict.items() if k.startswith('dqn.')}
        if not brain_sd or not dqn_sd:
            raise ValueError(
                f"'model' key found but missing 'brain.' or 'dqn.' prefixes. "
                f"Found {len(brain_sd)} brain keys and {len(dqn_sd)} dqn keys. "
                f"Sample keys: {list(state_dict.keys())[:5]}"
            )
    else:
        raise ValueError(
            f"Unrecognized checkpoint format. "
            f"Expected 'mortal'+'current_dqn' or 'model' keys. "
            f"Found keys: {list(checkpoint.keys())[:10]}"
        )

    print(f"Brain keys: {len(brain_sd)}, DQN keys: {len(dqn_sd)}")

    # Build models
    brain = Brain(conv_channels=args.conv_channels, num_blocks=args.num_blocks, version=version)
    dqn = DQN(version=version)

    brain.load_state_dict(brain_sd)
    dqn.load_state_dict(dqn_sd)

    # Critical: set eval mode for BatchNorm running stats
    brain.eval()
    dqn.eval()

    wrapper = PolicyONNXWrapper(brain, dqn)
    wrapper.eval()

    # Validate wrapper matches separate brain+dqn
    print("\nValidating wrapper against separate brain+dqn...")
    for batch_size in [1, 2, 4]:
        obs = torch.randn(batch_size, channels, width, dtype=torch.float32)
        mask = torch.ones(batch_size, 46, dtype=torch.float32)
        # Set some mask entries to 0
        mask[:, 37:] = 0.0  # disable riichi and above
        mask[:, 0] = 1.0    # enable discard 1m

        with torch.no_grad():
            phi = brain(obs)
            ref_out = dqn(phi, mask.to(torch.bool))
            wrap_out = wrapper(obs, mask)

        # Compare only finite values (masked actions are -inf; -inf - -inf = NaN)
        finite = torch.isfinite(ref_out) & torch.isfinite(wrap_out)
        if finite.any():
            diff = (ref_out[finite] - wrap_out[finite]).abs().max().item()
        else:
            diff = 0.0
        print(f"  batch={batch_size}: max_diff={diff:.2e}", end="")
        if diff < 1e-6:
            print(" OK")
        else:
            print(f" WARNING: large difference!")

    # Export to ONNX
    print(f"\nExporting to {args.output_path} (float32)")
    dummy_obs = torch.randn(2, channels, width, dtype=torch.float32)
    dummy_mask = torch.ones(2, 46, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_mask),
        args.output_path,
        input_names=['obs', 'mask'],
        output_names=['q_values'],
        dynamic_axes={
            'obs': {0: 'batch_size'},
            'mask': {0: 'batch_size'},
            'q_values': {0: 'batch_size'},
        },
        opset_version=17,
    )

    # Verify with onnx checker
    print("Verifying exported ONNX model structure...")
    import onnx
    model = onnx.load(args.output_path)
    onnx.checker.check_model(model)
    print(f"  ONNX model validated OK")
    print(f"  Inputs: {[(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in model.graph.input]}")
    print(f"  Outputs: {[(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in model.graph.output]}")

    # Verify with onnxruntime
    print("\nVerifying with onnxruntime...")
    import onnxruntime as ort
    session = ort.InferenceSession(args.output_path)

    for batch_size in [1, 2, 4]:
        obs = torch.randn(batch_size, channels, width, dtype=torch.float32)
        mask = torch.ones(batch_size, 46, dtype=torch.float32)
        mask[:, 37:] = 0.0

        with torch.no_grad():
            ref_out = wrapper(obs, mask).numpy()

        onnx_out = session.run(None, {
            'obs': obs.numpy(),
            'mask': mask.numpy(),
        })[0]

        # Compare only finite values (masked actions are -inf; -inf - -inf = NaN)
        ref_finite = np.isfinite(ref_out)
        onnx_finite = np.isfinite(onnx_out)
        both_finite = ref_finite & onnx_finite
        if both_finite.any():
            diff = np.abs(ref_out[both_finite] - onnx_out[both_finite]).max()
        else:
            diff = 0.0
        # Non-finite positions should match (both -inf where masked)
        inf_match = np.array_equal(ref_finite, onnx_finite)
        print(f"  batch={batch_size}: PyTorch vs ONNX max_diff={diff:.2e}, inf_match={inf_match}", end="")
        if diff < 1e-5 and inf_match:
            print(" OK")
        else:
            print(f" WARNING: large difference or inf mismatch!")

    file_size = os.path.getsize(args.output_path)
    print(f"\nDone! ONNX model saved to {args.output_path} ({file_size / 1024 / 1024:.1f} MB)")


if __name__ == '__main__':
    main()
