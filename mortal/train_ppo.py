import multiprocessing; multiprocessing.set_start_method("spawn", force=True)
def train():
    import prelude

    import logging
    import sys
    import os
    import gc
    import shutil
    import random
    import numpy as np
    import torch
    from copy import deepcopy
    from os import path
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from torch.distributions import Categorical
    from common import parameter_count, tqdm
    from player import TestPlayer, TrainPlayer
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
    num_workers = config['dataset']['num_workers']
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']

    # Policy config
    policy_cfg = config['policy']
    clip_ratio = policy_cfg['clip_ratio']
    dual_clip = policy_cfg['dual_clip']
    entropy_weight = policy_cfg.get('entropy_weight', 0.05)

    # PPO config
    ppo_cfg = config.get('ppo', {})
    num_iters = ppo_cfg.get('num_iters', 500)
    eval_every = ppo_cfg.get('eval_every', 50)
    ppo_epochs = ppo_cfg.get('ppo_epochs', 3)
    target_kl = ppo_cfg.get('target_kl', 0.015)

    # Dynamic entropy targeting config (Suphx Equation 3)
    ent_cfg = config.get('entropy_targeting', {})
    ent_targeting_enabled = ent_cfg.get('enabled', False)
    ent_target = ent_cfg.get('target', 2.0)
    ent_adapt_rate = ent_cfg.get('adapt_rate', 0.01)
    ent_weight_min = ent_cfg.get('weight_min', 0.01)
    ent_weight_max = ent_cfg.get('weight_max', 0.5)
    ent_floor = ent_cfg.get('floor', 0.5)

    norm = config['resnet'].get('norm', 'GN')  # Default to GN for online training
    mortal = Brain(version=version, norm=norm, **{k: v for k, v in config['resnet'].items() if k != 'norm'}).to(device)
    policy_net = CategoricalPolicy().to(device)

    # Freeze Brain: PPO only trains the 274K policy head, not the 21M Brain
    mortal.eval()
    for param in mortal.parameters():
        param.requires_grad = False

    all_models = (policy_net,)
    if enable_compile:
        mortal.compile()
        for m in all_models:
            m.compile()

    logging.info(f'version: {version}')
    logging.info(f'norm: {norm}')
    logging.info(f'obs shape: {obs_shape(version)}')
    brain_params = sum(p.numel() for p in mortal.parameters())
    logging.info(f'brain FROZEN: {brain_params:,} params (no gradients)')
    logging.info(f'policy_net TRAINABLE: {parameter_count(policy_net):,} params')

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
    train_player = TrainPlayer()
    best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    }

    steps = 0
    ppo_iter = 0
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        if 'policy_net' in state:
            policy_net.load_state_dict(state['policy_net'])
        else:
            logging.warning('no policy_net in checkpoint — starting from orthogonal init')
        is_ppo_checkpoint = 'ppo_iter' in state
        if 'optimizer' in state:
            if is_ppo_checkpoint:
                try:
                    optimizer.load_state_dict(state['optimizer'])
                    scheduler.load_state_dict(state['scheduler'])
                    logging.info(f'restored optimizer/scheduler from PPO checkpoint')
                except ValueError:
                    logging.warning('optimizer state incompatible (param group size changed) — using fresh optimizer')
            else:
                logging.info(f'skipping optimizer/scheduler from non-PPO checkpoint (fresh optimizer for PPO)')
        scaler.load_state_dict(state['scaler'])
        best_perf = state.get('best_perf', best_perf)
        if is_ppo_checkpoint:
            steps = state.get('steps', 0)
            ppo_iter = state.get('ppo_iter', 0)
            if 'entropy_weight' in state:
                entropy_weight = state['entropy_weight']
                logging.info(f'restored entropy_weight={entropy_weight:.6f}')
        else:
            steps = 0
            ppo_iter = 0
            logging.info(f'reset steps/ppo_iter to 0 for PPO transition')

    optimizer.zero_grad(set_to_none=True)

    # Old policy for importance sampling ratio (Brain is frozen, only policy_net changes)
    Old_policy_net = deepcopy(policy_net).eval()

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    writer = SummaryWriter(config['control']['tensorboard_dir'])

    def save_state():
        state = {
            'mortal': mortal.state_dict(),
            'policy_net': policy_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'steps': steps,
            'ppo_iter': ppo_iter,
            'entropy_weight': entropy_weight,
            'timestamp': datetime.now().timestamp(),
            'best_perf': best_perf,
            'config': config,
        }
        torch.save(state, state_file)

    def run_test_play():
        stat = test_player.test_play(test_games // 4, mortal, policy_net, device)
        policy_net.train()  # mortal stays eval (frozen)

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
            save_state()
            logging.info(
                'a new record has been made, '
                f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}, '
                f'saving to {best_state_file}'
            )
            shutil.copy(state_file, best_state_file)

    for iteration in range(ppo_iter, num_iters):
        ppo_iter = iteration
        logging.info(f'=== PPO iteration {iteration} ===')

        # Sync old policy to current before self-play (ensures ratio starts at 1.0)
        Old_policy_net.load_state_dict(policy_net.state_dict())

        # --- Self-play phase ---
        logging.info('generating self-play games...')
        policy_net.eval()  # mortal is always eval (frozen)
        rankings, file_list = train_player.train_play(mortal, policy_net, device)
        policy_net.train()  # mortal stays eval (frozen)

        rankings_np = np.array(rankings)
        avg_rank = rankings_np @ np.arange(1, 5) / rankings_np.sum()
        avg_pt = rankings_np @ np.array([90, 45, 0, -135]) / rankings_np.sum()
        logging.info(f'self-play rankings: {rankings_np} (rank={avg_rank:.4f}, pt={avg_pt:.2f})')
        writer.add_scalar('self_play/avg_rank', avg_rank, steps)
        writer.add_scalar('self_play/avg_pt', avg_pt, steps)

        # --- Training phase ---
        # Create shared stats for this iteration (persists across epochs for consistent normalization)
        manager = multiprocessing.Manager()
        shared_stats = {
            'count': manager.Value('d', 0.0),
            'mean': manager.Value('d', 0.0),
            'M2': manager.Value('d', 0.0),
            'lock': manager.Lock(),
        }

        iter_stats = {
            'ppo_loss': 0,
            'entropy': 0,
            'ratio_mean': 0,
            'ratio_max': 0,
            'approx_kl': 0,
            'batch_count': 0,
        }

        def train_batch(obs, actions, masks, advantage):
            nonlocal steps

            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            advantage = advantage.to(dtype=torch.float32, device=device)
            bs = obs.shape[0]
            assert masks[range(bs), actions].all()

            # Brain is frozen — compute features once, no gradient needed
            with torch.no_grad():
                with torch.autocast(device.type, enabled=enable_amp):
                    phi = mortal(obs)

            # Old policy log probs (old policy is frozen for entire iteration)
            with torch.no_grad():
                with torch.autocast(device.type, enabled=enable_amp):
                    old_probs = Old_policy_net(phi, masks)
                    old_dist = Categorical(probs=old_probs)
                    old_log_prob = old_dist.log_prob(actions)

            # Current policy (gradients flow through policy_net only)
            with torch.autocast(device.type, enabled=enable_amp):
                probs = policy_net(phi, masks)
                dist = Categorical(probs=probs)
                new_log_prob = dist.log_prob(actions)

                ratio = (new_log_prob - old_log_prob).exp()
                loss1 = ratio * advantage
                loss2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
                min_loss = torch.min(loss1, loss2)

                # Dual clipping: floor when advantage < 0
                clip_loss = torch.where(
                    advantage < 0,
                    torch.max(min_loss, dual_clip * advantage),
                    min_loss
                )

                entropy = dist.entropy().mean()
                entropy_loss = entropy * entropy_weight

                loss = -(clip_loss.mean() + entropy_loss)

            scaler.scale(loss / opt_step_every).backward()

            with torch.inference_mode():
                iter_stats['ppo_loss'] += loss.item()
                iter_stats['entropy'] += entropy.item()
                iter_stats['ratio_mean'] += ratio.mean().item()
                iter_stats['ratio_max'] = max(iter_stats['ratio_max'], ratio.max().item())
                approx_kl = ((ratio - 1) - (ratio.log())).mean().item()
                iter_stats['approx_kl'] += approx_kl
                iter_stats['batch_count'] += 1

            steps += 1
            if steps % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # Multi-epoch PPO: train on each batch of self-play data multiple times
        for epoch in range(ppo_epochs):
            # Shuffle file order each epoch for diversity
            epoch_files = file_list.copy()
            random.shuffle(epoch_files)

            file_data = FileDatasetsIter(
                version = version,
                file_list = epoch_files,
                pts = pts,
                file_batch_size = file_batch_size,
                player_names = ['trainee'],
                num_epochs = 1,
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
                persistent_workers = False,
                worker_init_fn = worker_init_fn,
            ))

            remaining_obs = []
            remaining_actions = []
            remaining_masks = []
            remaining_advantages = []
            remaining_bs = 0

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

            # KL early stopping: if policy has diverged too far, skip remaining epochs
            bc = max(iter_stats['batch_count'], 1)
            avg_kl = iter_stats['approx_kl'] / bc
            logging.info(f'  epoch {epoch}: avg_kl={avg_kl:.6f}, batches={iter_stats["batch_count"]}')
            if avg_kl > target_kl and epoch < ppo_epochs - 1:
                logging.info(f'  KL early stopping at epoch {epoch + 1}/{ppo_epochs} (avg_kl={avg_kl:.6f} > {target_kl})')
                break

        # Log iteration stats
        bc = max(iter_stats['batch_count'], 1)
        writer.add_scalar('loss/ppo_loss', iter_stats['ppo_loss'] / bc, steps)
        writer.add_scalar('entropy/entropy', iter_stats['entropy'] / bc, steps)
        writer.add_scalar('important_ratio/ratio', iter_stats['ratio_mean'] / bc, steps)
        writer.add_scalar('important_ratio/ratio_max', iter_stats['ratio_max'], steps)
        writer.add_scalar('important_ratio/approx_kl', iter_stats['approx_kl'] / bc, steps)
        writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
        writer.add_scalar('ppo/epochs_used', epoch + 1, steps)

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

        # Dynamic entropy targeting (Suphx Equation 3: alpha += beta * (H_target - H_bar))
        avg_entropy = iter_stats['entropy'] / bc
        if ent_targeting_enabled:
            old_weight = entropy_weight
            entropy_error = ent_target - avg_entropy
            entropy_weight += ent_adapt_rate * entropy_error
            entropy_weight = max(ent_weight_min, min(ent_weight_max, entropy_weight))
            if abs(entropy_weight - old_weight) > 1e-6:
                logging.info(f'  entropy targeting: {avg_entropy:.4f} vs target {ent_target:.2f}, '
                             f'weight {old_weight:.4f} -> {entropy_weight:.4f}')
            writer.add_scalar('entropy/target', ent_target, steps)
            writer.add_scalar('entropy/weight', entropy_weight, steps)

            # Entropy floor with rollback (safety net)
            if avg_entropy < ent_floor:
                logging.warning(f'ENTROPY COLLAPSE DETECTED: {avg_entropy:.4f} < floor {ent_floor}')
                logging.warning(f'Rolling back to old policy, doubling entropy weight')
                policy_net.load_state_dict(Old_policy_net.state_dict())
                entropy_weight = min(entropy_weight * 2.0, ent_weight_max)
                logging.warning(f'Entropy weight after doubling: {entropy_weight:.4f}')
                writer.add_scalar('entropy/rollback', 1, steps)

        writer.flush()
        logging.info(
            f'iter {iteration}: steps={steps}, '
            f'loss={iter_stats["ppo_loss"]/bc:.4f}, '
            f'entropy={iter_stats["entropy"]/bc:.4f}, '
            f'ratio={iter_stats["ratio_mean"]/bc:.4f}, '
            f'approx_kl={iter_stats["approx_kl"]/bc:.6f}, '
            f'ent_w={entropy_weight:.4f}, '
            f'epochs={epoch+1}/{ppo_epochs}, '
            f'batches={bc}'
        )

        # Save checkpoint every iteration
        save_state()

        # Periodic evaluation
        if (iteration + 1) % eval_every == 0:
            logging.info(f'running evaluation at iteration {iteration + 1}...')
            run_test_play()
            # BUG: CUDA hang after eval in online mode — restart workaround
            sys.exit(0)

        gc.collect()

def main():
    import os
    import sys
    import time
    from subprocess import Popen
    from config import config

    online = config['control'].get('online', False)
    if not online:
        # PPO always uses subprocess restart for the CUDA eval hang bug
        pass

    # do not set this env manually
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    if os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    cmd = (sys.executable, __file__)
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    while True:
        child = Popen(
            cmd,
            stdin = sys.stdin,
            stdout = sys.stdout,
            stderr = sys.stderr,
            env = env,
        )
        if (code := child.wait()) != 0:
            sys.exit(code)
        time.sleep(3)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
