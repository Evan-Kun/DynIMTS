import torch
import torch.nn as nn
from einops import rearrange

from .spatial_conv import SpatialConvOrderK
from .gcrnn import GCGRUCell, DGCGRUCell
from .spatial_attention import SpatialAttention
from ..utils.ops import reverse_tensor

import numpy as np


class Encoder(nn.Module):
    def __init__(self, d_in, d_model, d_out, support_len, order=1, attention_block=False, nheads=2, dropout=0.):
        super(Encoder, self).__init__()
        self.order = order
        self.lin_in = nn.Conv1d(d_in, d_model, kernel_size=1)
        self.adj = None
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin_in.weight)

    def forward(self, x, m, h, u, adj, cached_support=False):
        # [batch, channels, nodes]
        x_in = x
        x_in = self.lin_in(x_in)
        support = adj[0]
        support = support / (support.sum(1, keepdims=True) + 0.000001)

        if x_in.dim() < 4:
            squeeze = True
            x_0 = x_in.unsqueeze(-1)
        else:
            squeeze = False
        x_1 = torch.einsum('ncvl,wv->ncwl', (x_0, support)).contiguous()
        x_2 = torch.einsum('ncvl,wv->ncwl', (x_1, support)).contiguous()
        out = x_2
        if squeeze:
            out = out.squeeze(-1)
        out = self.dropout(out)
        return out


class RaindropLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 u_size=None,
                 n_layers=1,
                 dropout=0.,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 support_len=2,
                 n_nodes=None,
                 layer_norm=False,
                 n_class=2):
        super(RaindropLayer, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.u_size = int(u_size) if u_size is not None else 0
        self.n_layers = int(n_layers)
        # input + mask + (eventually) exogenous
        rnn_input_size = 2 * self.input_size + self.u_size
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(nn.GRUCell(input_size=n_nodes * self.hidden_size,
                              hidden_size=n_nodes * self.hidden_size))
            # self.cells.append(nn.LSTMCell(input_size=n_nodes * self.hidden_size,
            #                   hidden_size=n_nodes * self.hidden_size))
            if layer_norm:
                self.norms.append(nn.GroupNorm(
                    num_groups=1, num_channels=self.hidden_size))
            else:
                self.norms.append(nn.Identity())
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # Spatial decoder (rnn_input_size + hidden_size -> hidden_size)
        self.encoder = Encoder(d_in=self.input_size,
                               d_model=self.hidden_size,
                               d_out=self.input_size,
                               support_len=1,
                               order=decoder_order,
                               attention_block=global_att)

        # Hidden state initialization embedding
        if n_nodes is not None:
            self.h0 = self.init_hidden_states(n_nodes)
        else:
            self.register_parameter('h0', None)

        self.activation = nn.LeakyReLU()
        # self.activation = nn.ReLU()
        self.transformA = nn.Linear(n_nodes, n_nodes, bias=True)
        self.attention = nn.MultiheadAttention(n_nodes, 1)
        self.adj_attention = nn.MultiheadAttention(embed_dim=16, num_heads=1)

        self.n_class = n_class
        self.classifier = nn.Sequential(
            nn.Linear(n_nodes * self.hidden_size, 300),
            nn.ReLU(),
            # nn.Linear(300, 300),
            # nn.ReLU(),
            nn.Linear(300, self.n_class)
        )


        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                # nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight)

    def init_hidden_states(self, n_nodes):
        h0 = []
        for l in range(self.n_layers):
            std = 1. / \
                torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
            vals = torch.distributions.Normal(
                0, std).sample((1, self.hidden_size * n_nodes))
            vals = torch.zeros(1, self.hidden_size * n_nodes)
            xavier_norm = torch.nn.init.xavier_normal(
                torch.empty(1, self.hidden_size * n_nodes))
            kaiming_norm = torch.nn.init.kaiming_normal_(
                torch.empty(1, self.hidden_size * n_nodes), mode='fan_out')
            h0.append(nn.Parameter(vals))
            # h0.append(nn.Parameter(kaiming_norm))
        return nn.ParameterList(h0)

    def get_h0(self, x):
        if self.h0 is not None:
            observe = self.h0
            h0 = [h.expand(x.shape[0], -1) for h in self.h0]
            return h0
           # return [h.expand(x.shape[0], -1, -1) for h in self.h0]
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device)] * self.n_layers

    def update_state(self, x, h, adj):
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer]))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h

    def forward(self, x, adj, mask=None, u=None, h=None, cached_support=False):
        adj = [adj, adj]
        # x:[batch, features, nodes, steps]
        x = x[:, :, :adj[0].shape[0], :]
        mask = mask[:, :, :adj[0].shape[0], :]
        *_, steps = x.size()

        # infer all valid if mask is None
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)

        # init hidden state using node embedding or the empty state
        if h is None:
            h = self.get_h0(x)
        elif not isinstance(h, list):
            h = [*h]

        # Temporal conv
        predictions, imputations, states = [], [], []
        representations = []
        for step in range(steps):
            x_s = x[..., step]
            m_s = mask[..., step]
            h_s = h[-1]
            u_s = u[..., step] if u is not None else None
            diag = 1 - torch.eye(x_s.shape[2]).to(x.device)
            attention_adj = False
            # retrieve maximum information from neighbors
            repr_s = self.encoder(x=x_s, m=m_s, h=h_s, u=u_s, adj=adj,
                                  cached_support=cached_support)  # receive messages from neighbors (no self-loop!)
            # readout of imputation state + mask to retrieve imputations
            # prepare inputs

            repr_s = repr_s.view(repr_s.shape[0], -1)

            inputs = [repr_s]
            if u_s is not None:
                inputs.append(u_s)
            inputs = torch.cat(inputs, dim=1)  # x_hat_2 + mask + exogenous
            # update state with original sequence filled using imputations

            h = self.update_state(inputs, h, adj)

            # store imputations and states
            states.append(torch.stack(h, dim=0))
            representations.append(repr_s)
            if step == steps - 1:
                # torch.set_printoptions(threshold=10_000, linewidth=300)
                # print('learned graph: ')
                # print(adj_act_ms)
                h_enc = h[-1]

        # Aggregate outputs -> [batch, features, nodes, steps]
        states = torch.stack(states, dim=-1)
        representations = torch.stack(representations, dim=-1)
        h_final = h_enc.reshape(h_enc.shape[0], -1)
        output = self.classifier(h_final)
        # representations_output = self.classifier(representations)

        return representations, states, h_enc, output
