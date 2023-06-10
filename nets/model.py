import copy
from typing import Optional, Any, Union, Callable
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


from nets.tramsformer import MyBatchNorm1d, TransformerEncoderLayer, TransformerEncoder
from nets.Bresson2021TheTN import Transformer_decoder_net


def get_mask_indexes(nums, graph_size):
    """
    :param nums: (batch_size, 1)
    :param graph_size: torch.int
    :return: (batch_size, 2 * graph_size)
    """
    return torch.cat([
        torch.arange(graph_size) * graph_size + nums,
        nums * graph_size + torch.arange(graph_size)
    ], dim=1)


def get_mask_indexes_1d(template_idxes, graph_size, target_batches):
    """
    :param template_idxes: (batch_size, 2 * graph_size)
    :param graph_size: torch.int
    :param target_batches: (?)
    :return: (batch_size * 2 * graph_size)
    """
    return (template_idxes + graph_size * graph_size * target_batches.view(-1, 1).expand(-1, 2 * graph_size)).view(-1)


class Model(nn.Module):
    def __init__(
            self, graph_size: int, graph_dim: int = 2, d_model: int = 128, nhead: int = 8,
            num_encoder_layers: int = 3, num_decoder_layers: int = 2,
            dim_feedforward: int = 512, degree_constrain: int = None, is_stp: bool = False,
            dropout: float = 0.0, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            norm_eps: float = 1e-5, norm_first: bool = False, batch_first: bool = True,
            encoder_layer_norm: bool = False, device=None, dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Model, self).__init__()
        self.graph_dim = graph_dim
        self.d_model = d_model
        sqrt_d_model = int(math.sqrt(d_model))
        self.sqrt_d_model = sqrt_d_model

        self.non_terminal_emb = nn.Linear(graph_dim, d_model, **factory_kwargs)
        self.terminal_emb = nn.Linear(graph_dim, d_model, **factory_kwargs)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, norm_eps, batch_first, norm_first, **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=norm_eps, **factory_kwargs) if encoder_layer_norm else MyBatchNorm1d(d_model, eps=norm_eps, **factory_kwargs)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.linear_e = nn.Linear(2 * d_model, sqrt_d_model, **factory_kwargs)  # edge
        self.linear_k = nn.Linear(sqrt_d_model, num_decoder_layers * sqrt_d_model, **factory_kwargs)  # key
        self.linear_v = nn.Linear(sqrt_d_model, (num_decoder_layers - 1) * sqrt_d_model, **factory_kwargs)  # value
        self.linear_g = nn.Linear(d_model, sqrt_d_model, **factory_kwargs)  # graph

        self.decoder = Transformer_decoder_net(sqrt_d_model, nhead, num_decoder_layers, **factory_kwargs)

        # self._reset_parameters()
        self.is_stp = is_stp
        self.num_decoder_layers = num_decoder_layers
        self.degree_constrain = degree_constrain
        self.mask_2d21d_idxes = get_mask_indexes(torch.arange(graph_size).view(-1, 1), graph_size).to(device)
        # e.g. graph_size = 3, self.mask_2d21d_idxes:
        # tensor([[0, 3, 6, 0, 1, 2],
        #         [1, 4, 7, 3, 4, 5],
        #         [2, 5, 8, 6, 7, 8]])
        # self.alpha = alpha
        # self.alpha_decay = 0.95

    def forward(self, x, terminal_size, batch_size, greedy=True, return_weights=False, max_sample_size=None):
        """
        @param x: (batch_size, graph_size, graph_dim)
        @param terminal_size: int in [2, graph_size]
        @param batch_size: batch size, determining sample_size with terminal_size
        @param greedy: bool
        @param return_weights: bool
        @param max_sample_size: if x.size(0) is too small, make sure sample_size is not too big
        @return:
        """
        # parameters
        d_model = self.d_model
        sqrt_d_model = self.sqrt_d_model
        dataset_size, graph_size, graph_dim = x.size()  # dataset_size is batch_size // terminal_size in training, but may be 1 in testing
        if not self.is_stp:
            assert terminal_size == graph_size, 'graph_size not match with terminal_size, self.is_stp: {}, terminal_size: {}, graph_size: {}'.format(
                self.is_stp, terminal_size, graph_size
            )
        if not max_sample_size:
            max_sample_size = batch_size
        sample_size = min(batch_size // dataset_size, max_sample_size)  # smaller sample_size for single instance with large graph_size
        assert sample_size > 0, 'sample_size error, sample_size: {} = batch_size: {} // dataset_size: {}'.format(sample_size, batch_size, dataset_size)
        zero_to_bs = torch.arange(dataset_size * sample_size)
        replacement = False if sample_size <= terminal_size else True
        assert 2 <= terminal_size <= graph_size, 'size error, expect 2 <= terminal_size: {} <= graph_size: {}'.format(
            terminal_size, graph_size
        )

        embedding = torch.cat([
            self.terminal_emb(x[:, :terminal_size]),
            self.non_terminal_emb(x[:, terminal_size:])
        ], dim=1)  # (dataset_size, graph_size, d_model), vertex features
        v_encoder = self.encoder(embedding)  # (dataset_size, graph_size, d_model), vertex features

        e_encoder = torch.cat([
            v_encoder.unsqueeze(dim=2) - v_encoder.unsqueeze(dim=1),
            torch.max(v_encoder.unsqueeze(dim=2), v_encoder.unsqueeze(dim=1))
        ], dim=3)  # (dataset_size, graph_size, graph_size, d_model * 2)
        e_encoder = self.linear_e(e_encoder).view(dataset_size, graph_size * graph_size, sqrt_d_model).unsqueeze(1).expand(-1, sample_size, -1, -1).clone().view(
            dataset_size * sample_size, graph_size * graph_size, sqrt_d_model
        )  # (dataset_size * sample_size, graph_size * graph_size, sqrt_d_model)

        k_att_decoder = self.linear_k(e_encoder)  # (dataset_size * sample_size, graph_size * graph_size, num_decoder_layers * sqrt_d_model)
        v_att_decoder = self.linear_v(e_encoder)  # (dataset_size * sample_size, graph_size * graph_size, num_decoder_layers * sqrt_d_model)

        v_mean = self.linear_g(torch.mean(v_encoder, dim=1)).unsqueeze(dim=1).expand(-1, sample_size, -1).clone().view(
            dataset_size * sample_size, sqrt_d_model
        )  # (dataset_size * sample_size, sqrt_d_model), graph feature

        # initialization
        nodes = torch.multinomial(torch.full((terminal_size,), 1/terminal_size), num_samples=sample_size, replacement=replacement).to(x.device)  # note terminal nodes in stp are in front of other nodes
        nodes = nodes.unsqueeze(dim=0).expand(dataset_size, -1).clone().view(-1)  # (dataset_size * num_terminals)
        edges = torch.zeros([dataset_size * sample_size, 2, graph_size - 1], dtype=torch.long, device=x.device)  # node number: terminal_size - 1 <= graph_size -1
        degrees = torch.zeros([dataset_size * sample_size, graph_size], dtype=torch.int, device=x.device)  # for dcmst
        s = e_encoder[zero_to_bs, nodes * graph_size + nodes]  # (dataset_size * sample_size, sqrt_d_model)

        # create mask, suppose we choose left_nodes as first nodes, right_nodes as second nodes
        masks = torch.diag(
            torch.ones(graph_size, dtype=torch.int, device=x.device)
        ).repeat(dataset_size * sample_size, 1, 1)  # (dataset_size * sample_size, graph_size, graph_size)
        masks += 1  # mask all edges
        masks[zero_to_bs, nodes] -= 1  # open first nodes to be chosen as left nodes
        masks[zero_to_bs, :, nodes] += 1  # mask first nodes from being chosen as right nodes

        log_prob_sum = []
        weights_by_step = []
        alive = torch.ones(dataset_size * sample_size, dtype=torch.bool, device=x.device)  # (dataset_size * sample_size)
        lengths = torch.ones(dataset_size * sample_size, dtype=torch.int, device=x.device)  # (dataset_size * sample_size)
        self.decoder.reset_selfatt_keys_values()
        for step in range(graph_size - 1):
            h = torch.nn.MaxPool1d(2)(torch.cat([v_mean, s], dim=1))  # (dataset_size * sample_size, sqrt_d_model), partial solution features
            weights_mask = masks.view(dataset_size * sample_size, -1) > 0  # (dataset_size * sample_size, graph_size * graph_size)
            weights = self.decoder(h, k_att_decoder, v_att_decoder, weights_mask)  # (dataset_size * sample_size, graph_size * graph_size)

            if return_weights:
                weights_by_step.append(weights)
            if greedy:
                idxes = torch.argmax(weights, dim=1)  # (dataset_size * sample_size)
            else:
                idxes = Categorical(weights).sample()  # (dataset_size * sample_size)
            log_prob_sum.append(torch.log(weights[zero_to_bs, idxes]) * alive)
            left_nodes = torch.div(idxes, graph_size, rounding_mode='trunc').view(dataset_size * sample_size)
            right_nodes = (idxes % graph_size).view(dataset_size * sample_size)
            assert torch.all(left_nodes - right_nodes)  # left and right nodes must not be same
            degrees[zero_to_bs, left_nodes] += 1
            degrees[zero_to_bs, right_nodes] += 1  # update degree after an edge is added
            edges[zero_to_bs, 0, step] = left_nodes * alive
            edges[zero_to_bs, 1, step] = right_nodes * alive
            s = e_encoder[zero_to_bs, left_nodes * graph_size + right_nodes]

            # update mask
            masks[zero_to_bs, right_nodes] -= 1  # open newly visited nodes to be chosen as left nodes
            masks[zero_to_bs, :, right_nodes] += 1  # mask newly visited nodes from being chosen as right nodes

            if self.degree_constrain:
                masks = masks.view(-1)

                left_target_batches = torch.where(degrees[zero_to_bs, left_nodes] == self.degree_constrain)[0]
                left_idxes = get_mask_indexes_1d(self.mask_2d21d_idxes[left_nodes[left_target_batches]], graph_size, left_target_batches)
                masks[left_idxes] += 1

                right_target_batches = torch.where(degrees[zero_to_bs, right_nodes] == self.degree_constrain)[0]
                right_idxes = get_mask_indexes_1d(self.mask_2d21d_idxes[right_target_batches[right_target_batches]], graph_size, right_target_batches)
                masks[right_idxes] += 1

                masks = masks.view(dataset_size * sample_size, graph_size, graph_size)
            alive = alive * ~torch.all(degrees[zero_to_bs, :terminal_size], dim=1)  # if alive is 0, then the sequence is end
            lengths = lengths + alive  # (dataset_size * sample_size), range in [0, graph_size]
            if not torch.any(alive):  # if not any sequence is alive
                break

        log_prob_sum = torch.stack(log_prob_sum, dim=1).sum(dim=1)  # (dataset_size * sample_size)
        # edges: (dataset_size * sample_size, 2, graph_size), log_prob_sum, lengths: (dataset_size * sample_size)
        return edges, log_prob_sum, lengths, weights_by_step

