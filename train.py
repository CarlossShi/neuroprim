import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import numpy as np
import tqdm
import time
import os
import argparse
import math

from nets.model import Model
from utils import get_device, train_loop, validation_loop, get_costs, get_mrcst_costs


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--graph_size', type=int, default=20)  # 20, 50, 100
    parser.add_argument('--degree_constrain', type=int, default=None)  # None, 2
    parser.add_argument('--cost_function_key', type=str, default='default')  # default, mrcst
    parser.add_argument('--is_stp', type=bool, default=False)

    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)  # int(sqrt(d_model)) % nhead should be 0
    parser.add_argument('--num_encoder_layers', type=int, default=3)
    parser.add_argument('--num_decoder_layers', type=int, default=2)  # one enc-dec, one final
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=1)
    parser.add_argument('--baseline', type=str, default='pomo')  # pomo, None, rollout
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_batches', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=100)

    parser.add_argument('--wandb_project-name', type=str, default='neuro-prim-euclidean', help='the wandb\'s project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='the entity (team) of wandb\'s project')
    parser.add_argument('--track', action=argparse.BooleanOptionalAction)  # [Parsing boolean values with argparse](https://stackoverflow.com/a/15008806/12224183)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cost_functions = {'default': get_costs, 'mrcst': get_mrcst_costs}
    args = parse_args()
    torch.manual_seed(args.seed)

    # model parameters
    dtype = torch.float
    graph_size = args.graph_size
    graph_dim = 2
    d_model = args.d_model
    nhead = args.nhead
    assert not math.sqrt(d_model) % nhead, 'd_model not match with nhead, d_model: {}, nhead: {}'.format(d_model, nhead)
    num_encoder_layers = args.num_encoder_layers
    num_decoder_layers = args.num_decoder_layers
    dim_feedforward = args.dim_feedforward
    degree_constrain = args.degree_constrain
    is_stp = args.is_stp
    cost_function_key = args.cost_function_key
    cost_function = cost_functions[cost_function_key]

    # train parameters
    num_epochs = args.num_epochs
    num_batches = args.num_batches
    batch_size = args.batch_size
    run_name = time.strftime('%Y%m%d-%H%M%S_') + '_'.join([
        '{}{}'.format((k[0] + k[-1]) if (k_split := k.split('_')) == [k] else ''.join(_[0] for _ in k_split), v)
        for k, v in args.__dict__.items() if k not in [
            'track', 'wandb_project_name', 'wandb_entity'
        ]
    ])
    print('run name:', run_name)
    os.makedirs('checkpoints/{}'.format(run_name), exist_ok=True)
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    device, device_count = get_device()
    model_train = Model(
        graph_size, graph_dim, d_model,
        nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
        degree_constrain, is_stp,
        device=device, dtype=dtype
    ).to(device)
    optimizer = torch.optim.Adam(model_train.parameters(), lr=args.lr)
    writer = SummaryWriter('runs/{}'.format(run_name))

    with open('data/random/{}_test_seed1234.pkl'.format(graph_size), 'rb') as f:
        data = torch.tensor(np.load(f, allow_pickle=True), dtype=torch.float32).to(device)[:1]  # (batch_size, graph_size, dim)
    dataloader = DataLoader(data, batch_size=batch_size // graph_size)

    for epoch in tqdm.tqdm(range(1, num_epochs + 1), desc='outer loop', position=0):
        optimizer.param_groups[0]['lr'] *= args.lr_decay
        start_time = time.time()
        model_train.train()
        loss = train_loop(model_train, optimizer, num_batches, batch_size, graph_size, graph_dim, cost_function, device, dtype)
        model_train.eval()
        cost_train = validation_loop(model_train, dataloader, cost_function)
        print('epoch: {} loss: {:.2f} cost_train: {:.2f}, lr: {:.5f}'.format(epoch, loss, cost_train, optimizer.param_groups[0]['lr']))
        writer.add_scalar('Losses/training loss', loss, epoch)
        writer.add_scalar('Charts/validation cost', cost_train, epoch)
        writer.add_scalar('Charts/learning rate', optimizer.param_groups[0]['lr'], global_step=epoch)
        writer.add_scalar('Charts/epoch per second', epoch / (time.time() - start_time), global_step=epoch)
        torch.save(model_train.state_dict(), 'checkpoints/{}/{}.pt'.format(run_name, epoch))
        torch.cuda.empty_cache()
    writer.close()
