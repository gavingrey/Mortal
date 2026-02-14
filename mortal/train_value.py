import prelude

import random
import torch
import logging
from os import path
from glob import glob
from datetime import datetime
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from model import Brain, ValueNet
from common import tqdm
from config import config


class ValueFileDatasetsIter(IterableDataset):
    def __init__(self, version, file_list, file_batch_size=20, cycle=False):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.file_batch_size = file_batch_size
        self.cycle = cycle
        self.buffer = []
        self.iterator = None

    def build_iter(self):
        from libriichi.dataset import GameplayLoader
        loader = GameplayLoader(version=self.version, oracle=False)

        while True:
            random.shuffle(self.file_list)
            for start_idx in range(0, len(self.file_list), self.file_batch_size):
                batch_files = self.file_list[start_idx:start_idx + self.file_batch_size]
                self.populate_buffer(loader, batch_files)
                buffer_size = len(self.buffer)
                for i in random.sample(range(buffer_size), buffer_size):
                    yield self.buffer[i]
                self.buffer.clear()
            if not self.cycle:
                break

    def populate_buffer(self, loader, file_list):
        data = loader.load_gz_log_files(file_list)
        for file in data:
            for game in file:
                obs = game.take_obs()
                player_id = game.take_player_id()
                grp = game.take_grp()
                rank_by_player = grp.take_rank_by_player()
                final_rank = int(rank_by_player[player_id])

                for obs_i in obs:
                    self.buffer.append((obs_i, final_rank))

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def collate(batch):
    obs_list, rank_list = zip(*batch)
    obs = torch.stack([torch.as_tensor(o, dtype=torch.float32) for o in obs_list])
    ranks = torch.tensor(rank_list, dtype=torch.int64)
    return obs, ranks


def train():
    cfg = config['value']
    batch_size = cfg['control']['batch_size']
    save_every = cfg['control']['save_every']
    val_steps = cfg['control']['val_steps']

    device = torch.device(cfg['control']['device'])
    torch.backends.cudnn.benchmark = cfg['control']['enable_cudnn_benchmark']
    enable_amp = cfg['control'].get('enable_amp', True)

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    version = config['control']['version']

    # Load pre-trained Brain encoder (frozen)
    brain = Brain(version=version, **config['resnet']).to(device)
    mortal_state = torch.load(config['control']['state_file'], weights_only=True, map_location=device)
    brain.load_state_dict(mortal_state['mortal'])
    brain.eval()
    for param in brain.parameters():
        param.requires_grad = False
    logging.info('loaded pre-trained Brain encoder (frozen)')

    # Initialize ValueNet head
    value_net = ValueNet().to(device)
    optimizer = optim.AdamW(value_net.parameters(), lr=cfg['optim']['lr'])

    state_file = cfg['state_file']
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        value_net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        steps = state['steps']
    else:
        steps = 0

    # Build file index
    file_index = cfg['dataset']['file_index']
    train_globs = cfg['dataset']['train_globs']
    val_globs = cfg['dataset']['val_globs']
    if path.exists(file_index):
        index = torch.load(file_index, weights_only=True)
        train_file_list = index['train_file_list']
        val_file_list = index['val_file_list']
    else:
        logging.info('building file index...')
        train_file_list = []
        val_file_list = []
        for pat in train_globs:
            train_file_list.extend(glob(pat, recursive=True))
        for pat in val_globs:
            val_file_list.extend(glob(pat, recursive=True))
        train_file_list.sort(reverse=True)
        val_file_list.sort(reverse=True)
        torch.save({'train_file_list': train_file_list, 'val_file_list': val_file_list}, file_index)

    logging.info(f'train file list size: {len(train_file_list):,}')
    logging.info(f'val file list size: {len(val_file_list):,}')

    writer = SummaryWriter(cfg['control']['tensorboard_dir'])

    train_file_data = ValueFileDatasetsIter(
        version=version,
        file_list=train_file_list,
        file_batch_size=cfg['dataset']['file_batch_size'],
        cycle=True,
    )
    train_data_loader = iter(DataLoader(
        dataset=train_file_data,
        batch_size=batch_size,
        drop_last=True,
        num_workers=1,
        collate_fn=collate,
    ))

    val_file_data = ValueFileDatasetsIter(
        version=version,
        file_list=val_file_list,
        file_batch_size=cfg['dataset']['file_batch_size'],
        cycle=True,
    )
    val_data_loader = iter(DataLoader(
        dataset=val_file_data,
        batch_size=batch_size,
        drop_last=True,
        num_workers=1,
        collate_fn=collate,
    ))

    stats = {
        'train_loss': 0,
        'train_acc': 0,
        'val_loss': 0,
        'val_acc': 0,
    }

    approx_percent = steps * batch_size / (len(train_file_list) * 660) * 100
    logging.info(f'total steps: {steps:,} est. {approx_percent:6.3f}%')

    pb = tqdm(total=save_every, desc='TRAIN')
    for obs, ranks in train_data_loader:
        obs = obs.to(device=device)
        ranks = ranks.to(device=device)

        with torch.autocast(device.type, enabled=enable_amp):
            with torch.no_grad():
                phi = brain(obs)
            logits = value_net(phi)
            loss = F.cross_entropy(logits, ranks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.inference_mode():
            stats['train_loss'] += loss
            stats['train_acc'] += (logits.argmax(-1) == ranks).to(torch.float32).mean()

        steps += 1
        pb.update(1)

        if steps % save_every == 0:
            pb.close()

            with torch.inference_mode():
                value_net.eval()
                pb = tqdm(total=val_steps, desc='VAL')
                for idx, (obs, ranks) in enumerate(val_data_loader):
                    if idx == val_steps:
                        break
                    obs = obs.to(device=device)
                    ranks = ranks.to(device=device)

                    with torch.autocast(device.type, enabled=enable_amp):
                        phi = brain(obs)
                        logits = value_net(phi)
                        loss = F.cross_entropy(logits, ranks)

                    stats['val_loss'] += loss
                    stats['val_acc'] += (logits.argmax(-1) == ranks).to(torch.float32).mean()
                    pb.update(1)
                pb.close()
                value_net.train()

            writer.add_scalars('loss', {
                'train': stats['train_loss'] / save_every,
                'val': stats['val_loss'] / val_steps,
            }, steps)
            writer.add_scalars('acc', {
                'train': stats['train_acc'] / save_every,
                'val': stats['val_acc'] / val_steps,
            }, steps)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], steps)
            writer.flush()

            for k in stats:
                stats[k] = 0
            approx_percent = steps * batch_size / (len(train_file_list) * 660) * 100
            logging.info(f'total steps: {steps:,} est. {approx_percent:6.3f}%')

            state = {
                'model': value_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'steps': steps,
                'timestamp': datetime.now().timestamp(),
            }
            torch.save(state, state_file)
            pb = tqdm(total=save_every, desc='TRAIN')
    pb.close()


if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        pass
