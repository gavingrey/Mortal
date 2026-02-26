#!/usr/bin/env python3
"""
Convert a Brain checkpoint from BatchNorm to GroupNorm.

BN gamma/beta -> GN weight/bias (same shape, direct transfer).
BN running_mean/running_var are discarded (GN has none).

Usage:
    python scripts/convert_bn_to_gn.py <input.pth> <output.pth>
"""
import sys
import torch
from collections import OrderedDict


def convert_bn_to_gn(state_dict):
    """Convert Brain state dict keys from BatchNorm to GroupNorm format."""
    new_sd = OrderedDict()
    skipped = []

    for key, value in state_dict.items():
        # BN has: weight, bias, running_mean, running_var, num_batches_tracked
        # GN has: weight, bias
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            skipped.append(key)
            continue
        # BN weight/bias map directly to GN weight/bias (same shape)
        new_sd[key] = value

    return new_sd, skipped


def main():
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <input.pth> <output.pth>')
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    state = torch.load(input_path, weights_only=True, map_location='cpu')

    # Convert mortal (Brain) state dict
    mortal_sd = state['mortal']
    new_mortal_sd, skipped = convert_bn_to_gn(mortal_sd)

    print(f'Converted mortal state dict:')
    print(f'  Original keys: {len(mortal_sd)}')
    print(f'  New keys: {len(new_mortal_sd)}')
    print(f'  Skipped (BN-specific): {len(skipped)}')
    for key in skipped:
        print(f'    - {key}')

    # Update config to record norm change
    if 'config' in state:
        state['config']['resnet']['norm'] = 'GN'

    state['mortal'] = new_mortal_sd
    torch.save(state, output_path)
    print(f'\nSaved to {output_path}')
    print('Note: Create the Brain with norm="GN" when loading this checkpoint.')


if __name__ == '__main__':
    main()
