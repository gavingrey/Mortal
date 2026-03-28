import multiprocessing; multiprocessing.set_start_method("spawn", force=True)
def train():
    import prelude

    import logging
    import gc
    import gzip
    import json
    import shutil
    import random
    import sys
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

    def compute_oracle_gamma(step, warmup_steps, anneal_end_step):
        """Compute oracle dropout gamma: 1.0 during warmup, linear anneal to 0.0, then 0.0."""
        if step < warmup_steps:
            return 1.0
        if step >= anneal_end_step:
            return 0.0
        return 1.0 - (step - warmup_steps) / (anneal_end_step - warmup_steps)

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

    # AMP dtype: BF16 if available, else FP16
    amp_dtype_str = config['control'].get('amp_dtype', 'fp16')
    if amp_dtype_str == 'bf16':
        amp_dtype = torch.bfloat16
        # BF16 has full FP32 dynamic range, no GradScaler needed
        use_grad_scaler = False
    else:
        amp_dtype = torch.float16
        use_grad_scaler = True

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
    max_steps = config['optim']['scheduler']['max_steps']
    prefetch_factor = config['dataset'].get('prefetch_factor', 3)

    # Policy config
    policy_cfg = config['policy']
    awr_beta = policy_cfg['awr_beta']
    awr_clip = policy_cfg.get('awr_clip', 100)
    entropy_weight = policy_cfg.get('entropy_weight', 0.05)

    # Precomputed data config
    use_precomputed = config['dataset'].get('use_precomputed', False)
    precomputed_dir = config['dataset'].get('precomputed_dir', '')
    precomputed_rewards_file = config['dataset'].get('precomputed_rewards_file', '')
    gamma = config['dataset'].get('gamma', 1.0)

    oracle = config['dataset'].get('oracle', False)
    # Oracle dropout annealing config
    oracle_dropout_cfg = config.get('oracle_dropout', {})
    od_warmup_steps = oracle_dropout_cfg.get('warmup_steps', 0)
    od_anneal_end_step = oracle_dropout_cfg.get('anneal_end_step', 0)
    od_enabled = oracle and od_anneal_end_step > 0
    oracle_gamma = 1.0
    if od_enabled:
        logging.info(f'oracle dropout: warmup={od_warmup_steps}, anneal_end={od_anneal_end_step}')
    norm = config['resnet'].get('norm', 'BN')
    mortal = Brain(version=version, norm=norm, is_oracle=oracle, **{k: v for k, v in config['resnet'].items() if k != 'norm'}).to(device)
    policy_net = CategoricalPolicy().to(device)
    all_models = (mortal, policy_net)
    if enable_compile:
        for m in all_models:
            m.compile()

    logging.info(f'version: {version}')
    logging.info(f'norm: {norm}')
    logging.info(f'oracle: {oracle}')
    logging.info(f'amp_dtype: {amp_dtype_str}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'policy_net params: {parameter_count(policy_net):,}')
    if use_precomputed:
        logging.info(f'using precomputed shards from: {precomputed_dir}')
    elif precomputed_rewards_file:
        logging.info(f'using precomputed rewards from: {precomputed_rewards_file}')

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
    scaler = GradScaler(device.type, enabled=enable_amp and use_grad_scaler)
    test_player = TestPlayer()
    best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    }
    best_oracle_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    }

    steps = 0
    shuffle_seed = random.randint(0, 2**63)
    files_consumed = 0
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
        best_oracle_perf = state.get('best_oracle_perf', best_oracle_perf)
        steps = state.get('steps', 0)
        shuffle_seed = state.get('shuffle_seed', shuffle_seed)
        files_consumed = state.get('files_consumed', 0)
        if od_enabled and 'oracle_gamma' in state:
            logging.info(f'checkpoint oracle_gamma: {state["oracle_gamma"]:.4f}')

    optimizer.zero_grad(set_to_none=True)

    # Load precomputed rewards if configured
    precomputed_rewards = None
    if precomputed_rewards_file and path.exists(precomputed_rewards_file):
        logging.info(f'loading precomputed rewards from {precomputed_rewards_file}...')
        precomputed_rewards = torch.load(precomputed_rewards_file, weights_only=False, map_location='cpu')
        logging.info(f'precomputed rewards: {len(precomputed_rewards["rewards"]):,} files, '
                     f'mean={precomputed_rewards["global_mean"]:.6f}, '
                     f'std={precomputed_rewards["global_std"]:.6f}')

    # Create shared stats for Welford normalization (only when not using precomputed)
    shared_stats = None
    if not use_precomputed and precomputed_rewards is None:
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
        nonlocal files_consumed

        if use_precomputed:
            # Precomputed shard path — bypasses Rust FFI + GRP entirely
            scripts_dir = config['dataset'].get('scripts_dir', '')
            if scripts_dir and scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from precomputed_dataset import PrecomputedAWRDataset, precomputed_worker_init_fn

            # Load reward stats
            reward_stats_path = path.join(precomputed_dir, 'reward_stats.pt')
            if path.exists(reward_stats_path):
                reward_stats = torch.load(reward_stats_path, weights_only=True, map_location='cpu')
                global_mean = float(reward_stats['global_mean'])
                global_std = float(reward_stats['global_std'])
                logging.info(f'reward stats: mean={global_mean:.6f}, std={global_std:.6f}')
            else:
                logging.warning(f'no reward_stats.pt in {precomputed_dir}, using mean=0 std=1')
                global_mean = 0.0
                global_std = 1.0

            file_data = PrecomputedAWRDataset(
                shard_dir=precomputed_dir,
                oracle=oracle,
                gamma=gamma,
                global_mean=global_mean,
                global_std=global_std,
                suit_augment_mode=suit_augment_mode or 'random',
                num_epochs=num_epochs,
            )
            data_loader = iter(DataLoader(
                dataset=file_data,
                batch_size=batch_size,
                drop_last=enable_compile,  # avoid recompilation on remainder batches
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=num_workers > 0,
                worker_init_fn=precomputed_worker_init_fn,
            ))
            is_uint8_input = True
        else:
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

            # Deterministic shuffle for resumability
            rng = random.Random(shuffle_seed)
            rng.shuffle(file_list)

            # Skip already-processed files on resume
            if files_consumed > 0:
                if files_consumed >= len(file_list):
                    # All files consumed — start a new epoch with a fresh shuffle
                    epoch_num = files_consumed // len(file_list)
                    files_consumed = files_consumed % len(file_list)
                    logging.info(f'all files consumed (epoch {epoch_num}), re-shuffling for new pass, skipping {files_consumed:,} files')
                    rng2 = random.Random(shuffle_seed + epoch_num)
                    rng2.shuffle(file_list)
                if files_consumed > 0:
                    logging.info(f'resuming: skipping {files_consumed:,} already-processed files, {len(file_list) - files_consumed:,} remaining')
                    file_list = file_list[files_consumed:]

            file_data = FileDatasetsIter(
                version = version,
                file_list = file_list,
                pts = pts,
                oracle = oracle,
                file_batch_size = file_batch_size,
                reserve_ratio = reserve_ratio,
                player_names = player_names,
                num_epochs = num_epochs,
                enable_augmentation = enable_augmentation,
                augmented_first = augmented_first,
                suit_augment_mode = suit_augment_mode,
                pre_shuffled = True,
                policy_gradient = True,
                shared_stats = shared_stats,
                gamma = gamma,
                precomputed_rewards = precomputed_rewards,
            )
            data_loader = iter(DataLoader(
                dataset = file_data,
                batch_size = batch_size,
                drop_last = enable_compile,  # avoid recompilation on remainder batches
                num_workers = num_workers,
                pin_memory = True,
                prefetch_factor = prefetch_factor if num_workers > 0 else None,
                persistent_workers = num_workers > 0,
                worker_init_fn = worker_init_fn,
            ))
            is_uint8_input = False

        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

        remaining_obs = []
        remaining_invisible_obs = []
        remaining_actions = []
        remaining_masks = []
        remaining_advantages = []
        remaining_bs = 0
        pb = tqdm(total=save_every, desc='AWR', initial=steps % save_every)

        def train_batch(obs, invisible_obs, actions, masks, advantage):
            nonlocal steps
            nonlocal idx
            nonlocal pb

            # Update oracle dropout gamma
            if od_enabled:
                oracle_gamma = compute_oracle_gamma(steps, od_warmup_steps, od_anneal_end_step)

            # Handle uint8 inputs from precomputed shards
            if is_uint8_input and obs.dtype == torch.uint8:
                obs = obs.to(device=device).float() / 255.0
                if invisible_obs is not None:
                    invisible_obs = invisible_obs.to(device=device).float() / 255.0
            else:
                obs = obs.to(dtype=torch.float32, device=device)
                if invisible_obs is not None:
                    invisible_obs = invisible_obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            advantage = advantage.to(dtype=torch.float32, device=device)
            valid = masks[range(len(actions)), actions]
            if not valid.all():
                n_bad = (~valid).sum().item()
                logging.warning(f'skipping {n_bad}/{len(actions)} samples with invalid action-mask pairs')
                valid_idx = valid.nonzero(as_tuple=True)[0]
                obs = obs[valid_idx]
                if invisible_obs is not None:
                    invisible_obs = invisible_obs[valid_idx]
                actions = actions[valid_idx]
                masks = masks[valid_idx]
                advantage = advantage[valid_idx]
                if len(actions) == 0:
                    return

            # Oracle dropout: Bernoulli mask applied outside compiled graph
            if od_enabled and invisible_obs is not None and oracle_gamma < 1.0:
                mask = torch.bernoulli(
                    torch.full((obs.shape[0], 1, 1), oracle_gamma, device=obs.device)
                )
                invisible_obs = invisible_obs * mask

            with torch.autocast(device.type, dtype=amp_dtype, enabled=enable_amp):
                phi = mortal(obs, invisible_obs)
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
                if od_enabled:
                    writer.add_scalar('oracle_dropout/gamma', oracle_gamma, steps)

                # Log Welford stats (only in live pipeline mode)
                if shared_stats is not None:
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
                    'shuffle_seed': shuffle_seed,
                    'files_consumed': int(steps * batch_size / 1402),
                    'timestamp': datetime.now().timestamp(),
                    'best_perf': best_perf,
                    'best_oracle_perf': best_oracle_perf,
                    'config': config,
                    'oracle_gamma': oracle_gamma if od_enabled else 1.0,
                }
                torch.save(state, state_file)

                if steps % test_every == 0:
                    stat = test_player.test_play(test_games // 4, mortal, policy_net, device)
                    mortal.train()
                    policy_net.train()

                    avg_pt = stat.avg_pt([90, 45, 0, -135])
                    oracle_better = avg_pt >= best_oracle_perf['avg_pt'] and stat.avg_rank <= best_oracle_perf['avg_rank']
                    if oracle_better:
                        best_oracle_perf['avg_pt'] = avg_pt
                        best_oracle_perf['avg_rank'] = stat.avg_rank
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

                    # Zero-oracle evaluation (feature transfer diagnostic)
                    if oracle:
                        logging.info('running zero-oracle test play...')
                        stat_zo = test_player.test_play_zero_oracle(test_games // 4, mortal, policy_net, device)
                        mortal.train()
                        policy_net.train()

                        avg_pt_zo = stat_zo.avg_pt([90, 45, 0, -135])
                        logging.info(f'zero-oracle avg rank: {stat_zo.avg_rank:.6}')
                        logging.info(f'zero-oracle avg pt: {avg_pt_zo:.6}')
                        writer.add_scalar('test_play_zero_oracle/avg_ranking', stat_zo.avg_rank, steps)
                        writer.add_scalar('test_play_zero_oracle/avg_pt', avg_pt_zo, steps)
                        writer.add_scalars('test_play_zero_oracle/ranking', {
                            '1st': stat_zo.rank_1_rate,
                            '2nd': stat_zo.rank_2_rate,
                            '3rd': stat_zo.rank_3_rate,
                            '4th': stat_zo.rank_4_rate,
                        }, steps)
                        writer.flush()

                        # Track best by zero-oracle performance (what we deploy)
                        better = avg_pt_zo >= best_perf['avg_pt'] and stat_zo.avg_rank <= best_perf['avg_rank']
                        if better:
                            past_best = best_perf.copy()
                            best_perf['avg_pt'] = avg_pt_zo
                            best_perf['avg_rank'] = stat_zo.avg_rank

                    if not oracle:
                        better = avg_pt >= best_perf['avg_pt'] and stat.avg_rank <= best_perf['avg_rank']
                        if better:
                            past_best = best_perf.copy()
                            best_perf['avg_pt'] = avg_pt
                            best_perf['avg_rank'] = stat.avg_rank

                    if better:
                        torch.save(state, state_file)
                        logging.info(
                            'a new record has been made, '
                            f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                            f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}, '
                            f'saving to {best_state_file}'
                        )
                        shutil.copy(state_file, best_state_file)
                    if oracle_better:
                        oracle_best_file = best_state_file.replace('_best.', '_oracle_best.')
                        logging.info(
                            f'oracle record: rank {best_oracle_perf["avg_rank"]:.4}, '
                            f'pt {best_oracle_perf["avg_pt"]:.4}, saving to {oracle_best_file}'
                        )
                        shutil.copy(state_file, oracle_best_file)
                pb = tqdm(total=save_every, desc='AWR')

        for batch in data_loader:
            if oracle:
                obs, invisible_obs, actions, masks, advantage = batch
            else:
                obs, actions, masks, advantage = batch
                invisible_obs = None
            bs = obs.shape[0]
            if bs != batch_size:
                remaining_obs.append(obs)
                if invisible_obs is not None:
                    remaining_invisible_obs.append(invisible_obs)
                remaining_actions.append(actions)
                remaining_masks.append(masks)
                remaining_advantages.append(advantage)
                remaining_bs += bs
                continue
            train_batch(obs, invisible_obs, actions, masks, advantage)
            if steps >= max_steps:
                break

        remaining_batches = remaining_bs // batch_size
        if remaining_batches > 0:
            obs = torch.cat(remaining_obs, dim=0)
            invisible_obs = torch.cat(remaining_invisible_obs, dim=0) if remaining_invisible_obs else None
            actions = torch.cat(remaining_actions, dim=0)
            masks = torch.cat(remaining_masks, dim=0)
            advantage = torch.cat(remaining_advantages, dim=0)
            start = 0
            end = batch_size
            while end <= remaining_bs:
                train_batch(
                    obs[start:end],
                    invisible_obs[start:end] if invisible_obs is not None else None,
                    actions[start:end],
                    masks[start:end],
                    advantage[start:end],
                )
                if steps >= max_steps:
                    break
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
