import multiprocessing; multiprocessing.set_start_method("spawn", force=True)
def train():
    import prelude

    import logging
    import gc
    import gzip
    import json
    import shutil
    import random
    import torch
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from torch.distributions import Categorical
    from common import parameter_count, filtered_trimmed_lines, tqdm
    from player import TestPlayer
    from dataloader import FileDatasetsIter, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, CategoricalPolicy
    from libriichi.consts import obs_shape
    from config import config

    version = config['control']['version']

    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    test_games = config['test_play']['games']
    assert save_every % opt_step_every == 0
    assert test_every % save_every == 0

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    pts = config['env']['pts']
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    num_epochs = config['dataset']['num_epochs']
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']
    suit_augment_mode = config['dataset'].get('suit_augment_mode')
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']

    # Policy config
    policy_cfg = config['policy']
    awr_beta = policy_cfg['awr_beta']
    awr_clip = policy_cfg.get('awr_clip', 100)
    entropy_weight = policy_cfg.get('entropy_weight', 0.05)

    norm = config['resnet'].get('norm', 'BN')
    mortal = Brain(version=version, norm=norm, **{k: v for k, v in config['resnet'].items() if k != 'norm'}).to(device)
    policy_net = CategoricalPolicy().to(device)
    all_models = (mortal, policy_net)
    if enable_compile:
        for m in all_models:
            m.compile()

    logging.info(f'version: {version}')
    logging.info(f'norm: {norm}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'policy_net params: {parameter_count(policy_net):,}')

    decay_params = []
    no_decay_params = []
    for model in all_models:
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)
    test_player = TestPlayer()
    best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    }

    steps = 0
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        if 'policy_net' in state:
            policy_net.load_state_dict(state['policy_net'])
            logging.info('loaded policy_net from checkpoint')
        else:
            logging.info('no policy_net in checkpoint, using fresh orthogonal init')
        # Only load optimizer/scheduler if checkpoint has matching param groups
        # (DQN checkpoints have 3 param groups vs AWR's 2, causing a crash)
        is_awr_checkpoint = 'policy_net' in state
        if 'optimizer' in state and is_awr_checkpoint:
            try:
                optimizer.load_state_dict(state['optimizer'])
                scheduler.load_state_dict(state['scheduler'])
            except (ValueError, KeyError) as e:
                logging.warning(f'could not load optimizer state (param group mismatch?): {e}')
                logging.warning('starting optimizer from scratch')
        elif 'optimizer' in state:
            logging.info('skipping optimizer state from DQN checkpoint (incompatible param groups)')
        if 'scaler' in state:
            scaler.load_state_dict(state['scaler'])
        best_perf = state.get('best_perf', best_perf)
        steps = state.get('steps', 0)

    optimizer.zero_grad(set_to_none=True)

    # Create shared stats for Welford normalization across DataLoader workers
    manager = multiprocessing.Manager()
    shared_stats = {
        'count': manager.Value('d', 0.0),
        'mean': manager.Value('d', 0.0),
        'M2': manager.Value('d', 0.0),
        'lock': manager.Lock(),
    }

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'awr_loss': 0,
        'entropy': 0,
    }
    idx = 0

    def train_epoch():
        nonlocal steps
        nonlocal idx

        player_names_set = set()
        for filename in config['dataset']['player_names_files']:
            with open(filename) as f:
                player_names_set.update(filtered_trimmed_lines(f))
        player_names = list(player_names_set)
        logging.info(f'loaded {len(player_names):,} players')

        file_index = config['dataset']['file_index']
        if path.exists(file_index):
            index = torch.load(file_index, weights_only=True)
            file_list = index['file_list']
        else:
            logging.info('building file index...')
            file_list = []
            for pat in config['dataset']['globs']:
                file_list.extend(glob(pat, recursive=True))
            if len(player_names_set) > 0:
                filtered = []
                for filename in tqdm(file_list, unit='file'):
                    with gzip.open(filename, 'rt') as f:
                        start = json.loads(next(f))
                        if not set(start['names']).isdisjoint(player_names_set):
                            filtered.append(filename)
                file_list = filtered
            file_list.sort(reverse=True)
            torch.save({'file_list': file_list}, file_index)
        logging.info(f'file list size: {len(file_list):,}')

        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

        if num_workers > 1:
            random.shuffle(file_list)
        file_data = FileDatasetsIter(
            version = version,
            file_list = file_list,
            pts = pts,
            file_batch_size = file_batch_size,
            reserve_ratio = reserve_ratio,
            player_names = player_names,
            num_epochs = num_epochs,
            enable_augmentation = enable_augmentation,
            augmented_first = augmented_first,
            suit_augment_mode = suit_augment_mode,
            policy_gradient = True,
            shared_stats = shared_stats,
        )
        data_loader = iter(DataLoader(
            dataset = file_data,
            batch_size = batch_size,
            drop_last = False,
            num_workers = num_workers,
            pin_memory = True,
            prefetch_factor = 3 if num_workers > 0 else None,
            persistent_workers = num_workers > 0,
            worker_init_fn = worker_init_fn,
        ))

        remaining_obs = []
        remaining_actions = []
        remaining_masks = []
        remaining_advantages = []
        remaining_bs = 0
        pb = tqdm(total=save_every, desc='AWR', initial=steps % save_every)

        def train_batch(obs, actions, masks, advantage):
            nonlocal steps
            nonlocal idx
            nonlocal pb

            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            advantage = advantage.to(dtype=torch.float32, device=device)
            assert masks[range(batch_size), actions].all()

            with torch.autocast(device.type, enabled=enable_amp):
                phi = mortal(obs)
                probs = policy_net(phi, masks)
                dist = Categorical(probs=probs)
                log_prob = dist.log_prob(actions)

                exp_adv = torch.exp(advantage / awr_beta)
                if awr_clip is not None:
                    exp_adv = torch.clamp(exp_adv, max=awr_clip)

                awr_loss = -(exp_adv * log_prob).mean()

                entropy = dist.entropy().mean()
                entropy_loss = -entropy_weight * entropy

                loss = awr_loss + entropy_loss

            scaler.scale(loss / opt_step_every).backward()

            with torch.inference_mode():
                stats['awr_loss'] += awr_loss.item()
                stats['entropy'] += entropy.item()

            steps += 1
            idx += 1
            if idx % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            pb.update(1)

            if steps % save_every == 0:
                pb.close()

                writer.add_scalar('loss/awr_loss', stats['awr_loss'] / save_every, steps)
                writer.add_scalar('entropy/entropy', stats['entropy'] / save_every, steps)
                writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)

                # Log Welford stats
                with shared_stats['lock']:
                    w_count = shared_stats['count'].value
                    w_mean = shared_stats['mean'].value
                    w_M2 = shared_stats['M2'].value
                if w_count > 0:
                    w_std = (w_M2 / w_count) ** 0.5
                    writer.add_scalar('stats/count', w_count, steps)
                    writer.add_scalar('stats/mean', w_mean, steps)
                    writer.add_scalar('stats/std_dev', w_std, steps)

                writer.flush()

                for k in stats:
                    stats[k] = 0
                idx = 0

                before_next_test_play = (test_every - steps % test_every) % test_every
                logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

                state = {
                    'mortal': mortal.state_dict(),
                    'policy_net': policy_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'steps': steps,
                    'timestamp': datetime.now().timestamp(),
                    'best_perf': best_perf,
                    'config': config,
                }
                torch.save(state, state_file)

                if steps % test_every == 0:
                    stat = test_player.test_play(test_games // 4, mortal, policy_net, device)
                    mortal.train()
                    policy_net.train()

                    avg_pt = stat.avg_pt([90, 45, 0, -135])
                    better = avg_pt >= best_perf['avg_pt'] and stat.avg_rank <= best_perf['avg_rank']
                    if better:
                        past_best = best_perf.copy()
                        best_perf['avg_pt'] = avg_pt
                        best_perf['avg_rank'] = stat.avg_rank

                    logging.info(f'avg rank: {stat.avg_rank:.6}')
                    logging.info(f'avg pt: {avg_pt:.6}')
                    writer.add_scalar('test_play/avg_ranking', stat.avg_rank, steps)
                    writer.add_scalar('test_play/avg_pt', avg_pt, steps)
                    writer.add_scalars('test_play/ranking', {
                        '1st': stat.rank_1_rate,
                        '2nd': stat.rank_2_rate,
                        '3rd': stat.rank_3_rate,
                        '4th': stat.rank_4_rate,
                    }, steps)
                    writer.add_scalars('test_play/behavior', {
                        'agari': stat.agari_rate,
                        'houjuu': stat.houjuu_rate,
                        'fuuro': stat.fuuro_rate,
                        'riichi': stat.riichi_rate,
                    }, steps)
                    writer.add_scalars('test_play/agari_point', {
                        'overall': stat.avg_point_per_agari,
                        'riichi': stat.avg_point_per_riichi_agari,
                        'fuuro': stat.avg_point_per_fuuro_agari,
                        'dama': stat.avg_point_per_dama_agari,
                    }, steps)
                    writer.add_scalar('test_play/houjuu_point', stat.avg_point_per_houjuu, steps)
                    writer.add_scalar('test_play/point_per_round', stat.avg_point_per_round, steps)
                    writer.add_scalars('test_play/key_step', {
                        'agari_jun': stat.avg_agari_jun,
                        'houjuu_jun': stat.avg_houjuu_jun,
                        'riichi_jun': stat.avg_riichi_jun,
                    }, steps)
                    writer.add_scalars('test_play/riichi', {
                        'agari_after_riichi': stat.agari_rate_after_riichi,
                        'houjuu_after_riichi': stat.houjuu_rate_after_riichi,
                        'chasing_riichi': stat.chasing_riichi_rate,
                        'riichi_chased': stat.riichi_chased_rate,
                    }, steps)
                    writer.add_scalar('test_play/riichi_point', stat.avg_riichi_point, steps)
                    writer.add_scalars('test_play/fuuro', {
                        'agari_after_fuuro': stat.agari_rate_after_fuuro,
                        'houjuu_after_fuuro': stat.houjuu_rate_after_fuuro,
                    }, steps)
                    writer.add_scalar('test_play/fuuro_num', stat.avg_fuuro_num, steps)
                    writer.add_scalar('test_play/fuuro_point', stat.avg_fuuro_point, steps)
                    writer.flush()

                    if better:
                        torch.save(state, state_file)
                        logging.info(
                            'a new record has been made, '
                            f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                            f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}, '
                            f'saving to {best_state_file}'
                        )
                        shutil.copy(state_file, best_state_file)
                pb = tqdm(total=save_every, desc='AWR')

        for obs, actions, masks, advantage in data_loader:
            bs = obs.shape[0]
            if bs != batch_size:
                remaining_obs.append(obs)
                remaining_actions.append(actions)
                remaining_masks.append(masks)
                remaining_advantages.append(advantage)
                remaining_bs += bs
                continue
            train_batch(obs, actions, masks, advantage)

        remaining_batches = remaining_bs // batch_size
        if remaining_batches > 0:
            obs = torch.cat(remaining_obs, dim=0)
            actions = torch.cat(remaining_actions, dim=0)
            masks = torch.cat(remaining_masks, dim=0)
            advantage = torch.cat(remaining_advantages, dim=0)
            start = 0
            end = batch_size
            while end <= remaining_bs:
                train_batch(
                    obs[start:end],
                    actions[start:end],
                    masks[start:end],
                    advantage[start:end],
                )
                start = end
                end += batch_size
        pb.close()

    while True:
        train_epoch()
        gc.collect()
        # only run one epoch for offline for easier control
        break

if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        pass
