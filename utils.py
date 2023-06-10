import torch
from torch.distributions.categorical import Categorical
import tqdm
import numpy as np


def get_device():
    print('torch.cuda.is_available():', torch.cuda.is_available())
    device_count = torch.cuda.device_count()
    print('torch.cuda.device_count():', device_count)
    device_idxes = list(range(device_count))
    print('device_idxes:', device_idxes)
    devices = [torch.cuda.device(_) for _ in device_idxes]
    print('devices:', devices)
    device_names = [torch.cuda.get_device_name(_) for _ in device_idxes]
    print('device_names:', device_names, '\n')

    current_device = torch.cuda.current_device()
    print('torch.cuda.current_device():', current_device)
    print('torch.cuda.device(current_device):', devices[current_device])
    print('torch.cuda.get_device_name(current_device):', device_names[current_device], '\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, '\n')

    if device.type == 'cuda':
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    return device, device_count


def edges2adjlist(edges):
    """
    @param edges: (graph_size - 1, 2), np.array
    """
    graph_size = len(edges) + 1
    result = {_: set() for _ in range(graph_size)}
    for (l, r) in edges:
        result[l].add(r)
        result[r].add(l)
    return result


def f(last_node, curr_node, x, adj_list, graph_size):
    """

    @param last_node: int
    @param curr_node: int
    @param x: (graph_size, 2)
    @param adj_list: dict
    @param graph_size: int
    @return: int, float
    """
    total_count = 1
    total_cost = 0
    if last_node and len(adj_list[curr_node]) == 1:  # leaf node
        return 1, np.linalg.norm(x[last_node] - x[curr_node]) * (graph_size - 1)
    for _ in adj_list[curr_node]:
        if _ != last_node:
            count, cost = f(curr_node, _, x, adj_list, graph_size)
            total_count += count
            total_cost += cost
    if last_node is not None:
        total_cost += np.linalg.norm(x[last_node]-x[curr_node]) * total_count * (graph_size - total_count)
    return total_count, total_cost


def get_costs(x, edges):
    """
    @param x: (batch_size, graph_size, 2)
    @param edges: (batch_size, 2, graph_size - 1)
    @return: (batch_size)
    """
    batch_size, graph_size, _ = x.shape
    assert edges.shape[-1] == graph_size - 1
    temp = torch.gather(
        x, 1, edges.view(batch_size, -1).unsqueeze(-1).expand(batch_size, 2 * (graph_size - 1), 2)
    ).view(batch_size, 2, graph_size - 1, 2)
    return (temp[:, 0] - temp[:, 1]).norm(p=2, dim=2).sum(1)


def get_costs_from_D(D, edges):
    """
    @param D: (batch_size, graph_size, graph_size)
    @param edges: (batch_size, 2, graph_size - 1)
    @return: (batch_size)
    """
    batch_size, graph_size, graph_size2 = D.shape
    assert graph_size == graph_size2 == edges.shape[-1] + 1
    return torch.tensor(
        [sum(D[_][edges[_][0], edges[_][1]]) for _ in range(batch_size)]
    ).to(edges.device)


def get_mrcst_costs(x, edges):
    """
    cost function for minimum routing cost spanning tree
    @param x: (batch_size, graph_size, 2)
    @param edges: (batch_size, 2, graph_size - 1)
    @return: (batch_size)
    """
    x = x.cpu()
    batch_size, graph_size, _ = x.shape
    costs = torch.empty(batch_size, dtype=torch.float)
    for i in range(batch_size):
        adj_list = edges2adjlist(np.array(edges[i].cpu()).transpose())
        count, cost = f(None, 0, x[i], adj_list, graph_size)  # x[i] not x[0]!!
        assert count == graph_size
        costs[i] = cost
    return costs.to(edges.device)


def train_loop(model, optimizer, num_batches, batch_size, graph_size, graph_dim, cost_function, device, dtype):
    terminal_size = int(Categorical(torch.ones(graph_size - 1)).sample() + 2) if model.is_stp else graph_size
    total_loss = 0
    for step in tqdm.tqdm(range(num_batches), desc='inner loop', position=1, leave=False):
        data = torch.rand(batch_size // terminal_size, graph_size, graph_dim, device=device, dtype=dtype)
        edges, log_prob_sums, lengths, weights_by_step = model(
            data, terminal_size=terminal_size, batch_size=batch_size, greedy=False
        )  # edges: (len(data) * sample_size, 2, graph_size - 1); log_prob_sum, lengths: (len(data) * sample_size)
        assert not len(edges) % len(data), 'length error, len(edges): {}, len(data): {}'.format(len(edges), len(data))
        sample_size = len(edges) // len(data)
        costs = cost_function(
            data.unsqueeze(dim=1).expand(-1, sample_size, -1, -1).clone().view(len(data) * sample_size, graph_size, graph_dim),
            edges
        )  # (len(data) * sample_size)
        mean_cost = torch.mean(costs.view(len(data), sample_size), dim=1)  # (len(data))
        loss = torch.mean((costs - mean_cost.repeat_interleave(sample_size)) * log_prob_sums)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
    return total_loss / num_batches


def validation_loop(model, dataloader, cost_function):
    """
    quick validation with sample_size = 1
    @param model:
    @param dataloader:
    @param cost_function:
    @return: (float, float)
    """
    dataset_size, graph_size, graph_dim = dataloader.dataset.shape
    terminal_sizes = [2, graph_size // 2, graph_size] if model.is_stp else [graph_size]
    cost = 0
    with torch.no_grad():
        for terminal_size in terminal_sizes:
            for data in dataloader:  # (len(data), graph_size, 2)
                edges, _, _, _ = model(data, terminal_size=terminal_size, batch_size=len(data), greedy=True)  # (len(data), 2, graph_size - 1), if batch_size = len(data), then sample_size = 1
                assert not len(edges) % len(data), 'length error, len(edges): {}, len(data): {}'.format(len(data) * len(edges), len(data))
                sample_size = len(edges) // len(data)
                costs = cost_function(
                    data.unsqueeze(dim=1).expand(-1, sample_size, -1, -1).clone().view(len(data) * sample_size, graph_size, graph_dim),
                    edges
                )  # (len(data) * sample_size)
                batch_cost = torch.min(costs.view(-1, sample_size), dim=1)[0]  # (len(data))
                assert len(batch_cost) == len(data)
                cost += torch.sum(batch_cost)
    return cost / dataset_size / len(terminal_sizes)
