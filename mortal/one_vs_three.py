import prelude

import numpy as np
import torch
import secrets
import os
from model import Brain, DQN, CategoricalPolicy
from engine import MortalEngine
from libriichi.arena import OneVsThree
from config import config

def main():
    cfg = config['1v3']
    games_per_iter = cfg['games_per_iter']
    seeds_per_iter = games_per_iter // 4
    iters = cfg['iters']
    log_dir = cfg['log_dir']
    use_akochan = cfg['akochan']['enabled']

    if (key := cfg.get('seed_key', -1)) == -1:
        key = secrets.randbits(64)

    def load_engine(state, engine_cfg):
        cfg_inner = state['config']
        version = cfg_inner['control'].get('version', 1)
        conv_channels = cfg_inner['resnet']['conv_channels']
        num_blocks = cfg_inner['resnet']['num_blocks']
        norm = cfg_inner['resnet'].get('norm', 'BN')
        brain = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks, norm=norm).eval()
        brain.load_state_dict(state['mortal'])

        if 'policy_net' in state:
            head = CategoricalPolicy().eval()
            head.load_state_dict(state['policy_net'])
        else:
            head = DQN(version=version).eval()
            head.load_state_dict(state['current_dqn'])

        if engine_cfg['enable_compile']:
            brain.compile()
            head.compile()
        return MortalEngine(
            brain,
            head,
            is_oracle = False,
            version = version,
            device = torch.device(engine_cfg['device']),
            enable_amp = engine_cfg['enable_amp'],
            enable_rule_based_agari_guard = engine_cfg['enable_rule_based_agari_guard'],
            name = engine_cfg['name'],
        ), version

    if use_akochan:
        os.environ['AKOCHAN_DIR'] = cfg['akochan']['dir']
        os.environ['AKOCHAN_TACTICS'] = cfg['akochan']['tactics']
    else:
        state = torch.load(cfg['champion']['state_file'], weights_only=False, map_location=torch.device('cpu'))
        engine_cham, _ = load_engine(state, cfg['champion'])

    state = torch.load(cfg['challenger']['state_file'], weights_only=False, map_location=torch.device('cpu'))
    engine_chal, version = load_engine(state, cfg['challenger'])

    seed_start = 10000
    for i, seed in enumerate(range(seed_start, seed_start + seeds_per_iter * iters, seeds_per_iter)):
        print('-' * 50)
        print('#', i)
        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = log_dir,
        )
        if use_akochan:
            rankings = env.ako_vs_py(
                engine = engine_chal,
                seed_start = (seed, key),
                seed_count = seeds_per_iter,
            )
        else:
            rankings = env.py_vs_py(
                challenger = engine_chal,
                champion = engine_cham,
                seed_start = (seed, key),
                seed_count = seeds_per_iter,
            )
        rankings = np.array(rankings)
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ np.array([90, 45, 0, -135]) / rankings.sum()
        print(f'challenger rankings: {rankings} ({avg_rank}, {avg_pt}pt)')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
