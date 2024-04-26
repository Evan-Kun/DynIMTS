import torch
import torch.nn as nn
from einops import rearrange

from .spatial_conv import SpatialConvOrderK
from .gcrnn import GCGRUCell, DGCGRUCell
from .spatial_attention import SpatialAttention
from ..utils.ops import reverse_tensor

import numpy as np

class SpatialDecoder(nn.Module):
    def __init__(self, d_in, d_model, d_out, support_len, order=1, attention_block=False, nheads=2, dropout=0.):
        super(SpatialDecoder, self).__init__()
        self.order = order
        self.lin_in = nn.Conv1d(d_in, d_model, kernel_size=1)
        self.graph_conv = SpatialConvOrderK(c_in=d_model, c_out=d_model,
                                            support_len=support_len * order, order=1, include_self=False)
        if attention_block:
            self.spatial_att = SpatialAttention(d_in=d_model,
                                                d_model=d_model,
                                                nheads=nheads,
                                                dropout=dropout)
            self.lin_out = nn.Conv1d(3 * d_model, d_model, kernel_size=1)
        else:
            self.register_parameter('spatial_att', None)
            self.lin_out = nn.Conv1d(2 * d_model, d_model, kernel_size=1)
        self.read_out = nn.Conv1d(2 * d_model, d_out, kernel_size=1)
        self.activation = nn.PReLU()
        self.adj = None

    def forward(self, x, m, h, u, adj, cached_support=False):
        # [batch, channels, nodes]
        x_in = [x, m, h] if u is None else [x, m, u, h]
        x_in = torch.cat(x_in, 1)
        if self.order > 1:
            if cached_support and (self.adj is not None):
                adj = self.adj
            else:
                adj = SpatialConvOrderK.compute_support_orderK(adj, self.order, include_self=False, device=x_in.device)
                self.adj = adj if cached_support else None

        x_in = self.lin_in(x_in)
        out = self.graph_conv(x_in, adj)
        if self.spatial_att is not None:
            # [batch, channels, nodes] -> [batch, steps, nodes, features]
            x_in = rearrange(x_in, 'b f n -> b 1 n f')
            out_att = self.spatial_att(x_in, torch.eye(x_in.size(2), dtype=torch.bool, device=x_in.device))
            out_att = rearrange(out_att, 'b s n f -> b f (n s)')
            out = torch.cat([out, out_att], 1)
        out = torch.cat([out, h], 1)
        out = self.activation(self.lin_out(out))
        # out = self.lin_out(out)
        out = torch.cat([out, h], 1)
        return self.read_out(out), out


class GRIL(nn.Module):
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
                 layer_norm=False):
        super(GRIL, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.u_size = int(u_size) if u_size is not None else 0
        self.n_layers = int(n_layers)
        rnn_input_size = 2 * self.input_size + self.u_size  # input + mask + (eventually) exogenous

        # Spatio-temporal encoder (rnn_input_size -> hidden_size)
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(GCGRUCell(d_in=rnn_input_size if i == 0 else self.hidden_size,
                                        num_units=self.hidden_size, support_len=support_len, order=kernel_size))
            if layer_norm:
                self.norms.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size))
            else:
                self.norms.append(nn.Identity())
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # normlisation for adj
        self.adjnorm = nn.GroupNorm(num_groups=1, num_channels=n_nodes)

        # Fist stage readout
        self.first_stage = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.input_size, kernel_size=1)

        # Spatial decoder (rnn_input_size + hidden_size -> hidden_size)
        self.spatial_decoder = SpatialDecoder(d_in=rnn_input_size + self.hidden_size,
                                              d_model=self.hidden_size,
                                              d_out=self.input_size,
                                              support_len=2,
                                              order=decoder_order,
                                              attention_block=global_att)

        # Hidden state initialization embedding
        if n_nodes is not None:
            self.h0 = self.init_hidden_states(n_nodes)
            self.adj0 = self.init_adj(n_nodes)
            # self.s = self.h0

        else:
            self.register_parameter('h0', None)

        # learn a adj A
        self.activation = nn.Sigmoid()
        # self.activation = nn.LReLU()
        # self.activation = nn.ReLU()
        self.transformA = nn.Linear(n_nodes, n_nodes, bias=True)
        self.attention = nn.MultiheadAttention(n_nodes, 1)
        self.adj_attention = nn.MultiheadAttention(embed_dim=16, num_heads=1)


    def init_hidden_states(self, n_nodes):
        h0 = []
        for l in range(self.n_layers):
            std = 1. / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
            vals = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes))
            h0.append(nn.Parameter(vals))
        return nn.ParameterList(h0)

    def get_h0(self, x):
        if self.h0 is not None:
            h_test = [h for h in self.h0]
            h0 = [h.expand(x.shape[0], -1, -1) for h in self.h0]
            # h0 = [h.expand(x.shape[0], -1, -1) for h in self.s]
            return h0
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device)] * self.n_layers


    def init_adj(self, n_nodes):
        adj = []
        # val = torch.ones(n_nodes, n_nodes)
        val = torch.zeros(n_nodes, n_nodes)
        # adj.append(nn.Parameter(val))
        # return nn.ParameterList(adj)
        adj.append(val)
        return adj

    def get_adj(self, x):
        if self.adj0 is not None:
            # adj_test = [adj for adj in self.adj0]
            # adj0 = [adj.expand(x.shape[0], -1, -1) for adj in self.adj0]
            adj0 = [adj.to(x.device) for adj in self.adj0]
            return adj0
        # return [torch.ones(size=(x.shape[0], x.shape[2], x.shape[2])).to(x.device)] * self.n_layers
        return [torch.zeros(size=(x.shape[0], x.shape[2], x.shape[2])).to(x.device)] * self.n_layers

    def update_state(self, x, h, adj):
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer], adj))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h


    def update_adj(self, adj):
        # output = self.adjnorm(adj)
        # adj_act_std, adj_act_mean = torch.std_mean(output)
        # adj_act_ms = torch.exp(-torch.square((output - adj_act_mean) / adj_act_std))
        # adj_mask = adj_act_ms > 0.1
        # adj_mask = adj_mask.long()
        # adj_act_ms = adj_act_ms * adj_mask
        return adj

    #
    # def update_adj(self, h_s):
    #     self.s = h_s
    #     return self.s

    def forward(self, x, adj, mask=None, u=None, h=None, cached_support=False, adj_dynamic=None):
        # x:[batch, features, nodes, steps]
        *_, steps = x.size()

        # infer all valid if mask is None
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)

        # init hidden state using node embedding or the empty state
        if h is None:
            h = self.get_h0(x)
            adj_dynamic = self.get_adj(x)
            # s = self.get_h0(x)
            # print(adj_dynamic)
        elif not isinstance(h, list):
            print(h)
            h = [*h]
            print(h)
            print(adj_dynamic)
            adj_dynamic = [*adj_dynamic].to(x.device)
            print(adj_dynamic)


        # mean adj of this batch
        # adj_mean = torch.mean(x, 0, True)
        # adj_mean = torch.squeeze(adj_mean)
        # adj_mean = adj_mean.cpu().numpy()
        # adj_mean = np.corrcoef(adj_mean)
        # adj_mean = np.nan_to_num(adj_mean, nan=0).astype(np.float32)
        # adj_mean = torch.from_numpy(adj_mean).to(device='cuda')
        # print(adj_mean)

        # x_np = torch.squeeze(x).cpu().numpy()
        # adj_corr = []
        # for data in x_np:
        #     A = np.corrcoef(data)
        #     # MIN-MAX normalization
        #     np.fill_diagonal(A, 0.)
        #     A_mask = A > 0.1
        #     A_mask = A_mask.astype(int)
        #     A = A * A_mask
        #     A_sum = np.sum(A, axis=0, keepdims=True) + 0.00001
        #     A = A / A_sum
        #     adj_corr.append(A)
        # adj_corr = np.nan_to_num(adj_corr, nan=0).astype(np.float32)
        # adj_corr = torch.from_numpy(adj_corr).to(device='cuda')
        #
        # x_adj = torch.squeeze(x)
        # x_adj_n = x_adj / torch.norm(x_adj, dim=(1, 2)).view(32, 1, 1)
        # x_adj_t = x_adj.permute((0, 2, 1))
        # x_adj_t_n = x_adj_t / torch.norm(x_adj_t, dim=(1, 2)).view(32, 1, 1)
        # adj_A = torch.bmm(x_adj_n, x_adj_t_n)
        #
        # adj_dis = torch.cdist(x_adj_t, x_adj_t, p=2)
        #
        # adj_dig = torch.eye(x.shape[2]).repeat(x.shape[0], 1, 1).bool()
        # adj_A[adj_dig] = 0

        # Temporal conv
        predictions, imputations, states = [], [], []
        representations = []
        # adj_ini = torch.ones(x.shape[0], x.shape[2], x.shape[2]).cuda()
        # adjs = [adj_ini]
        adj_dis = 0
        # adj = [adj_mean, adj_mean]
        # adj = [adj[0].expand(x.shape[0], -1, -1), adj[1].expand(x.shape[0], -1, -1)]
        # adj = [adj_corr, adj_corr]
        # adj_0 = adj[0].unsqueeze(0).expand(x.shape[0], x.shape[2], x.shape[2])
        # adj_1 = adj[1].unsqueeze(0).expand(x.shape[0], x.shape[2], x.shape[2])
        # adj = [adj_0, adj_1]
        for step in range(steps):
            x_s = x[..., step]
            m_s = mask[..., step]
            h_s = h[-1]
            diag = 1 - torch.eye(x_s.shape[2]).to(x.device)
            # adj_s = adj_dynamic[-1] * diag
            # adj_s = self.adj0[-1].to(x.device)
            # if (step == 1):
            #     print('\n--------------------Step 1-----------------------')
                # print('\n--------------------Current A-----------------------')
                # print(adj_s)
                # print('\n-----------------------------------------------------------')
            # calculate A based on h
            # h_s_t = h_s.permute((0, 2, 1))
            # h_s_t_n = h_s_t / torch.norm(h_s_t, dim=(1, 2)).view(h_s.shape[0], 1, 1)
            # h_s_n = h_s / torch.norm(h_s, dim=(1, 2)).view(h_s.shape[0], 1, 1)
            # adj_h_n = torch.bmm(h_s_t_n, h_s_n)
            # adj = [adj_h_n, adj_h_n]
            # adj_A = torch.bmm(h_s_t, h_s)
            # # transform A via MLP
            # adj_A = self.transformA(adj_A)
            # adj_A = self.activation(adj_A)
            # if(step == 0):
            #     print('Step: ', step)
            #     adj = self.adj0
            #     print(h_s)
            #
            if (step != steps ):
                attention_adj = False
                # print('Step: ', step)



                if (attention_adj==True):

                    # attention
                    h_s_t = h_s.permute((2, 0, 1))
                    attn_output1, attn_output_weights1 = self.adj_attention(h_s_t, h_s_t, h_s_t)
                    adj_h_n = attn_output_weights1

                    # adj_h_n_sum = torch.sum(adj_h_n, 1)
                    # adj_h_n = adj_h_n / adj_h_n_sum
                    # attn_output, attn_output_weights = self.attention(adj_h_n_attention, adj_h_n_attention, adj_h_n_attention)
                    # adj_h_n_dia = adj_h_n.diagonal(dim1=1, dim2=2)
                    # adj_h_n_dia_unsq = torch.unsqueeze(adj_h_n_dia, dim=2)
                    # adj_h_n_dia_exp = adj_h_n_dia.expand(-1, -1, 36)
                    # adj_h_n_norm = adj_h_n / adj_h_n_dia_unsq
                    # adj_h_n_attention = adj_h_n_norm.permute((1, 0, 2))
                    adj_h_n_attention = adj_h_n.permute((1, 0, 2))
                    attn_output, attn_output_weights = self.attention(adj_h_n_attention, adj_h_n_attention, adj_h_n_attention)

                    adj_attention = torch.einsum('ncw,nwv->ncv', attn_output_weights, adj_h_n)
                    # adj_attention = torch.einsum('ncw,nwv->ncv', attn_output_weights, adj_h_n_norm)
                    adj_attention_mean = torch.mean(adj_attention, 0)
                    adj_h_n = adj_attention_mean
                    # print('\n--------------------Attention Mean A-----------------------')
                    # print(adj_attention_mean)
                    # print('\n-----------------------------------------------------------')

                    # print('\n------------------------cosine A---------------------------')
                    # print(adj_h_n)
                    # print('\n-----------------------------------------------------------')
                    adj_std, adj_mean = torch.std_mean(adj_h_n)
                    adj_h_n_ms = torch.exp(-torch.square((adj_h_n - adj_mean) / (adj_std + 0.000001)))
                    adj_mlp = self.transformA(adj_h_n_ms)
                    adj_act = self.activation(adj_mlp)

                    # normalisation
                    adj_act_std, adj_act_mean = torch.std_mean(adj_act)
                    adj_act_ms = torch.exp(-torch.square((adj_act - adj_act_mean) / adj_act_std))
                    adj_act_ms = adj_act_ms * diag
                    adj_mask = adj_act_ms > 0.1
                    adj_mask = adj_mask.long()
                    adj_act_ms = adj_act_ms * adj_mask

                    adj_sum = torch.sum(adj_act_ms, 1)
                    # adj_sum = adj_sum.expand(h_s.shape[0], h_s.shape[2], h_s.shape[2])
                    adj_result = adj_act_ms / (adj_sum + 0.000001)
                    # adj = [adj_result, adj_result]


                # if(attention_adj != True):
                #
                #     # cosine similarity
                #     h_s_t = h_s.permute((0, 2, 1))
                #     h_s_t_n = h_s_t / ( torch.norm(h_s_t, dim=(2), keepdim=True).expand(-1, -1, h_s_t.shape[2]) + 0.000001)
                #     h_s_n = h_s / (torch.norm(h_s, dim=(1), keepdim=True).expand(-1, h_s.shape[1], h_s.shape[2]) + 0.000001)
                #     adj_h_n = torch.bmm(h_s_t_n, h_s_n)
                #     adj_h_n = torch.squeeze(adj_h_n)
                #
                #     adj_h_n_attention = adj_h_n.permute((1, 0, 2))
                #     attn_output, attn_output_weights = self.attention(adj_h_n_attention, adj_h_n_attention,
                #                                                       adj_h_n_attention)
                #
                #     adj_attention = torch.einsum('ncw,nwv->ncv', attn_output_weights, adj_h_n)
                #     # adj_attention = torch.einsum('ncw,nwv->ncv', attn_output_weights, adj_h_n_norm)
                #     adj_attention_mean = torch.mean(adj_attention, 0)
                #     adj_h_n = adj_attention_mean
                #
                #     adj_std, adj_mean = torch.std_mean(adj_h_n)
                #     adj_h_n_ms = torch.exp(-torch.square((adj_h_n - adj_mean) / (adj_std + 0.000001)))
                #     adj_mlp = self.transformA(adj_h_n_ms)
                #     adj_act = self.activation(adj_mlp)
                #
                #     # normalisation
                #     adj_act_std, adj_act_mean = torch.std_mean(adj_act)
                #     adj_act_ms = torch.exp(-torch.square((adj_act - adj_act_mean) / adj_act_std))
                #     adj_act_ms = adj_act_ms * diag
                #
                #     # print('\n------------------normalisation MLP A----------------------')
                #     # print(adj_act_ms)
                #     # print('\n-----------------------------------------------------------')
                #
                #     # adj_act_norm = torch.norm(adj_act, dim=(1)).view(adj_act.shape[0], 1)
                #     # adj_act_n = adj_act / (adj_act_norm+0.000001)
                #     # adj_mask = adj_act > 0.05
                #     # adj_mask = adj_act > 0.1
                #     # adj_mask = adj_mask.long()
                #     # adj_act = adj_act * adj_mask
                #     # print('\n--------------------MLP            A-----------------------')
                #     # print(adj_act)
                #     # print('\n-----------------------------------------------------------')
                #     # alpha = 0.6(14.5), 0.8(15.8), 0.4(16.46) 0.5 (15.9) 0.7(14.68)
                #
                #     # alpha = 0.5
                #     # adj_alpha = alpha * adj_act_ms + (1 - alpha) * adj_s
                #     # adj_alpha_std, adj_alpha_mean = torch.std_mean(adj_alpha)
                #     # adj_alpha_ms = torch.exp(-torch.square((adj_alpha - adj_alpha_mean) / adj_alpha_std))
                #     # adj_act_ms = adj_alpha_ms
                #     # adj_mask = adj_alpha_ms > 0.1
                #     # adj_mask = adj_mask.long()
                #     # adj_alpha_ms = adj_alpha_ms * adj_mask
                #     # adj_sum = torch.sum(adj_alpha_ms, 1).unsqueeze(1)
                #     # adj_sum = adj_sum.expand(h_s.shape[0], h_s.shape[2], h_s.shape[2])
                #     # adj_result = adj_alpha_ms / (adj_sum + 0.000001
                #
                #     adj_mask = adj_act_ms > 0.1
                #     adj_mask = adj_mask.long()
                #     adj_act_ms = adj_act_ms * adj_mask
                #     adj_sum = torch.sum(adj_act_ms, 1)
                #     # adj_sum = adj_sum.expand(h_s.shape[0], h_s.shape[2], h_s.shape[2])
                #     adj_result = adj_act_ms / (adj_sum + 0.000001)
                #     adj = [adj_result, adj_result]
                # # else:
                #     # adj = [adj_h_n, adj_h_n]

            # adj_mask = adj_act_ms > 0.1
            # adj_mask = adj_mask.long()
            # adj_act_ms = adj_act_ms * adj_mask
            # print('\n------------------normalisation MLP A----------------------')
            # print(adj_act_ms)
            # print('\n-----------------------------------------------------------')
            # adj_sum = torch.sum(adj_act_ms, 1)
            # # adj_sum = torch.sum(adj_act, 1)
            # adj_sum = torch.sum(adj_act_ms, 1).unsqueeze(1)
            # adj_sum = adj_sum.expand(h_s.shape[0], h_s.shape[2], h_s.shape[2])
            # adj_result = adj_act_ms / (adj_sum + 0.000001)

            # for dgrin
            # adj = adj_result
                # print('\n------------------normalisation MLP A----------------------')
                # print(adj_act_ms)
                # print('\n-----------------------------------------------------------')
                # adj = [adj_result, adj_result]
                # adj_dynamic.append(adj_act_ms)
                # self.adj0.append(adj_act_ms)
            # else: adj = [adj_s, adj_s]

            #
            # if len(adjs) != 0:
            #     adj_dis += torch.sum(torch.abs(adj_result - adjs[-1]))
            # adjs.append(adj_result)
            # if (step == steps - 1):
            #     print('Step: ', step)
            #     # self.adj0 = adj_act_ms
            #     # self.h0 = h_s
            #     # self.s = self.update_adj(h_s)
            #     # print(h_s)
            #     print('\n------------------normalisation MLP A----------------------')
            #     print(adj_act_ms)
            #     print('\n-----------------------------------------------------------')
            # adj = [adj_corr, adj_corr]
            u_s = u[..., step] if u is not None else None
            # firstly impute missing values with predictions from state
            xs_hat_1 = self.first_stage(h_s)
            x_missing = x_s
            # fill missing values in input with prediction
            x_s = torch.where(m_s, x_s, xs_hat_1)
            # prepare inputs
            # retrieve maximum information from neighbors
            xs_hat_2, repr_s = self.spatial_decoder(x=x_s, m=m_s, h=h_s, u=u_s, adj=adj,
                                                    cached_support=cached_support)  # receive messages from neighbors (no self-loop!)
            # readout of imputation state + mask to retrieve imputations
            # prepare inputs
            x_s = torch.where(m_s, x_s, xs_hat_2)
            inputs = [x_s, m_s]
            if u_s is not None:
                inputs.append(u_s)
            inputs = torch.cat(inputs, dim=1)  # x_hat_2 + mask + exogenous
            # update state with original sequence filled using imputations
            h = self.update_state(inputs, h, adj)
            # adj_dynamic = self.update_adj(adj_act_ms)

            # store imputations and states
            imputations.append(xs_hat_2)
            predictions.append(xs_hat_1)
            states.append(torch.stack(h, dim=0))
            representations.append(repr_s)

        # Aggregate outputs -> [batch, features, nodes, steps]
        imputations = torch.stack(imputations, dim=-1)
        predictions = torch.stack(predictions, dim=-1)
        states = torch.stack(states, dim=-1)
        representations = torch.stack(representations, dim=-1)

        return imputations, predictions, representations, adj_dis, states
        # return imputations, predictions, representations, states


class BiGRIL(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size,
                 ff_dropout,
                 n_layers=1,
                 dropout=0.,
                 n_nodes=None,
                 support_len=2,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 u_size=0,
                 embedding_size=0,
                 layer_norm=False,
                 merge='mlp'):
        super(BiGRIL, self).__init__()
        self.fwd_rnn = GRIL(input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            dropout=dropout,
                            n_nodes=n_nodes,
                            support_len=support_len,
                            kernel_size=kernel_size,
                            decoder_order=decoder_order,
                            global_att=global_att,
                            u_size=u_size,
                            layer_norm=layer_norm)
        self.bwd_rnn = GRIL(input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            dropout=dropout,
                            n_nodes=n_nodes,
                            support_len=support_len,
                            kernel_size=kernel_size,
                            decoder_order=decoder_order,
                            global_att=global_att,
                            u_size=u_size,
                            layer_norm=layer_norm)

        if n_nodes is None:
            embedding_size = 0
        if embedding_size > 0:
            self.emb = nn.Parameter(torch.empty(embedding_size, n_nodes))
            nn.init.kaiming_normal_(self.emb, nonlinearity='relu')
        else:
            self.register_parameter('emb', None)

        if merge == 'mlp':
            self._impute_from_states = True
            self.out = nn.Sequential(
                nn.Conv2d(in_channels=4 * hidden_size + input_size + embedding_size,
                          out_channels=ff_size, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Conv2d(in_channels=ff_size, out_channels=input_size, kernel_size=1)
            )
        elif merge in ['mean', 'sum', 'min', 'max']:
            self._impute_from_states = False
            self.out = getattr(torch, merge)
        else:
            raise ValueError("Merge option %s not allowed." % merge)
        self.supp = None

    def forward(self, x, adj, mask=None, u=None, cached_support=False):
        if cached_support and (self.supp is not None):
            supp = self.supp
        else:
            supp = SpatialConvOrderK.compute_support(adj, x.device)
            self.supp = supp if cached_support else None
        # Forward
        fwd_out, fwd_pred, fwd_repr, _ = self.fwd_rnn(x, supp, mask=mask, u=u, cached_support=cached_support)
        # Backward
        rev_x, rev_mask, rev_u = [reverse_tensor(tens) for tens in (x, mask, u)]
        *bwd_res, _ = self.bwd_rnn(rev_x, supp, mask=rev_mask, u=rev_u, cached_support=cached_support)
        bwd_out, bwd_pred, bwd_repr = [reverse_tensor(res) for res in bwd_res]

        if self._impute_from_states:
            inputs = [fwd_repr, bwd_repr, mask]
            if self.emb is not None:
                b, *_, s = fwd_repr.shape  # fwd_h: [batches, channels, nodes, steps]
                inputs += [self.emb.view(1, *self.emb.shape, 1).expand(b, -1, -1, s)]  # stack emb for batches and steps
            imputation = torch.cat(inputs, dim=1)
            imputation = self.out(imputation)
        else:
            imputation = torch.stack([fwd_out, bwd_out], dim=1)
            imputation = self.out(imputation, dim=1)

        predictions = torch.stack([fwd_out, bwd_out, fwd_pred, bwd_pred], dim=0)

        return imputation, predictions



class DGRIL(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size,
                 ff_dropout,
                 n_layers=1,
                 dropout=0.,
                 n_nodes=None,
                 support_len=2,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 u_size=0,
                 embedding_size=0,
                 layer_norm=False,
                 merge='mlp'):
        super(DGRIL, self).__init__()
        self.fwd_rnn = GRIL(input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            dropout=dropout,
                            n_nodes=n_nodes,
                            support_len=support_len,
                            kernel_size=kernel_size,
                            decoder_order=decoder_order,
                            global_att=global_att,
                            u_size=u_size,
                            layer_norm=layer_norm)

        if n_nodes is None:
            embedding_size = 0
        if embedding_size > 0:
            self.emb = nn.Parameter(torch.empty(embedding_size, n_nodes))
            nn.init.kaiming_normal_(self.emb, nonlinearity='relu')
        else:
            self.register_parameter('emb', None)

        if merge == 'mlp':
            self._impute_from_states = True
            # Update for Dgrin
            self.out = nn.Sequential(
                nn.Conv2d(in_channels=2 * hidden_size + input_size + embedding_size,
                          out_channels=ff_size, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Conv2d(in_channels=ff_size, out_channels=input_size, kernel_size=1)
            )
        elif merge in ['mean', 'sum', 'min', 'max']:
            self._impute_from_states = False
            self.out = getattr(torch, merge)
        else:
            raise ValueError("Merge option %s not allowed." % merge)
        self.supp = None

    def forward(self, x, adj, mask=None, u=None, cached_support=False):
        if cached_support and (self.supp is not None):
            supp = self.supp
        else:
            supp = SpatialConvOrderK.compute_support(adj, x.device)
            self.supp = supp if cached_support else None
        # Forward
        fwd_out, fwd_pred, fwd_repr, adj_dis, _ = self.fwd_rnn(x, supp, mask=mask, u=u, cached_support=cached_support)
        # Backward
        # rev_x, rev_mask, rev_u = [reverse_tensor(tens) for tens in (x, mask, u)]
        # *bwd_res, _ = self.bwd_rnn(rev_x, supp, mask=rev_mask, u=rev_u, cached_support=cached_support)
        # bwd_out, bwd_pred, bwd_repr = [reverse_tensor(res) for res in bwd_res]

        if self._impute_from_states:
            inputs = [fwd_repr, mask]
            if self.emb is not None:
                b, *_, s = fwd_repr.shape  # fwd_h: [batches, channels, nodes, steps]
                inputs += [self.emb.view(1, *self.emb.shape, 1).expand(b, -1, -1, s)]  # stack emb for batches and steps
            imputation = torch.cat(inputs, dim=1)
            imputation = self.out(imputation)
        else:
            imputation = torch.stack([fwd_out], dim=1)
            imputation = self.out(imputation, dim=1)

        predictions = torch.stack([fwd_out, fwd_pred], dim=0)

        return imputation, predictions, adj_dis




class ODGRIL(nn.Module):
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
                 layer_norm=False):
        super(DGRIL, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.u_size = int(u_size) if u_size is not None else 0
        self.n_layers = int(n_layers)
        rnn_input_size = 2 * self.input_size + self.u_size  # input + mask + (eventually) exogenous

        # Spatio-temporal encoder (rnn_input_size -> hidden_size)
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(DGCGRUCell(d_in=rnn_input_size if i == 0 else self.hidden_size,
                                        num_units=self.hidden_size, support_len=support_len, order=kernel_size))
            if layer_norm:
                self.norms.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size))
            else:
                self.norms.append(nn.Identity())
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # Fist stage readout
        self.first_stage = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.input_size, kernel_size=1)

        # Spatial decoder (rnn_input_size + hidden_size -> hidden_size)
        self.spatial_decoder = SpatialDecoder(d_in=rnn_input_size + self.hidden_size,
                                              d_model=self.hidden_size,
                                              d_out=self.input_size,
                                              support_len=2,
                                              order=decoder_order,
                                              attention_block=global_att)

        # Hidden state initialization embedding
        if n_nodes is not None:
            self.h0 = self.init_hidden_states(n_nodes)
        else:
            self.register_parameter('h0', None)

    def init_hidden_states(self, n_nodes):
        h0 = []
        for l in range(self.n_layers):
            std = 1. / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
            vals = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes))
            h0.append(nn.Parameter(vals))
        return nn.ParameterList(h0)

    def get_h0(self, x):
        if self.h0 is not None:
            return [h.expand(x.shape[0], -1, -1) for h in self.h0]
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device)] * self.n_layers

    def update_state(self, x, h, adj):
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer], adj))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h

    def forward(self, x, adj, mask=None, u=None, h=None, cached_support=False):
        # x:[batch, features, nodes, steps]
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

        if cached_support and (self.supp is not None):
            supp = self.supp
        else:
            supp = SpatialConvOrderK.compute_dynamic_support(adj, x.device)
            self.supp = supp if cached_support else None

        for step in range(steps):
            x_s = x[..., step]
            m_s = mask[..., step]
            h_s = h[-1]
            u_s = u[..., step] if u is not None else None
            # firstly impute missing values with predictions from state
            xs_hat_1 = self.first_stage(h_s)
            # fill missing values in input with prediction
            x_s = torch.where(m_s, x_s, xs_hat_1)
            # prepare inputs
            # retrieve maximum information from neighbors
            xs_hat_2, repr_s = self.spatial_decoder(x=x_s, m=m_s, h=h_s, u=u_s, adj=supp,
                                                    cached_support=cached_support)  # receive messages from neighbors (no self-loop!)
            # readout of imputation state + mask to retrieve imputations
            # prepare inputs
            x_s = torch.where(m_s, x_s, xs_hat_2)
            inputs = [x_s, m_s]
            if u_s is not None:
                inputs.append(u_s)
            inputs = torch.cat(inputs, dim=1)  # x_hat_2 + mask + exogenous
            # update state with original sequence filled using imputations
            h, a = self.update_state(inputs, h, adj)
            # store imputations and states
            imputations.append(xs_hat_2)
            predictions.append(xs_hat_1)
            states.append(torch.stack(h, dim=0))
            representations.append(repr_s)

        # Aggregate outputs -> [batch, features, nodes, steps]
        imputations = torch.stack(imputations, dim=-1)
        predictions = torch.stack(predictions, dim=-1)
        states = torch.stack(states, dim=-1)
        representations = torch.stack(representations, dim=-1)

        return imputations, predictions, representations, states


class Encoder(nn.Module):
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
                 layer_norm=False):
        super(Encoder, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.u_size = int(u_size) if u_size is not None else 0
        self.n_layers = int(n_layers)
        rnn_input_size = 2 * self.input_size + self.u_size  # input + mask + (eventually) exogenous

        # Spatio-temporal encoder (rnn_input_size -> hidden_size)
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(GCGRUCell(d_in=rnn_input_size if i == 0 else self.hidden_size,
                                        num_units=self.hidden_size, support_len=support_len, order=kernel_size))
            if layer_norm:
                self.norms.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size))
            else:
                self.norms.append(nn.Identity())
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # Spatial decoder (rnn_input_size + hidden_size -> hidden_size)
        self.spatial_decoder = SpatialDecoder(d_in=rnn_input_size + self.hidden_size,
                                              d_model=self.hidden_size,
                                              d_out=self.input_size,
                                              support_len=2,
                                              order=decoder_order,
                                              attention_block=global_att)

        # Hidden state initialization embedding
        if n_nodes is not None:
            self.h0 = self.init_hidden_states(n_nodes)
        else:
            self.register_parameter('h0', None)

        # learn a adj A
        self.activation = nn.Sigmoid()
        # self.activation = nn.LReLU()
        # self.activation = nn.ReLU()
        self.transformA = nn.Linear(n_nodes, n_nodes, bias=True)
        self.attention = nn.MultiheadAttention(n_nodes, 1)
        self.adj_attention = nn.MultiheadAttention(embed_dim=16, num_heads=1)

    def init_hidden_states(self, n_nodes):
        h0 = []
        for l in range(self.n_layers):
            std = 1. / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
            vals = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes))
            h0.append(nn.Parameter(vals))
        return nn.ParameterList(h0)

    def get_h0(self, x):
        if self.h0 is not None:
            return [h.expand(x.shape[0], -1, -1) for h in self.h0]
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device)] * self.n_layers

    def update_state(self, x, h, adj):
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer], adj))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h

    def forward(self, x, adj, mask=None, u=None, h=None, cached_support=False):
        # x:[batch, features, nodes, steps]
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
            if(step != 0):
                if (attention_adj==True):

                    # attention
                    h_s_t = h_s.permute((2, 0, 1))
                    attn_output1, attn_output_weights1 = self.adj_attention(h_s_t, h_s_t, h_s_t)
                    adj_h_n = attn_output_weights1

                    # adj_h_n_sum = torch.sum(adj_h_n, 1)
                    # adj_h_n = adj_h_n / adj_h_n_sum
                    # attn_output, attn_output_weights = self.attention(adj_h_n_attention, adj_h_n_attention, adj_h_n_attention)
                    # adj_h_n_dia = adj_h_n.diagonal(dim1=1, dim2=2)
                    # adj_h_n_dia_unsq = torch.unsqueeze(adj_h_n_dia, dim=2)
                    # adj_h_n_dia_exp = adj_h_n_dia.expand(-1, -1, 36)
                    # adj_h_n_norm = adj_h_n / adj_h_n_dia_unsq
                    # adj_h_n_attention = adj_h_n_norm.permute((1, 0, 2))
                    adj_h_n_attention = adj_h_n.permute((1, 0, 2))
                    attn_output, attn_output_weights = self.attention(adj_h_n_attention, adj_h_n_attention, adj_h_n_attention)

                    adj_attention = torch.einsum('ncw,nwv->ncv', attn_output_weights, adj_h_n)
                    # adj_attention = torch.einsum('ncw,nwv->ncv', attn_output_weights, adj_h_n_norm)
                    adj_attention_mean = torch.mean(adj_attention, 0)
                    adj_h_n = adj_attention_mean
                    # print('\n--------------------Attention Mean A-----------------------')
                    # print(adj_attention_mean)
                    # print('\n-----------------------------------------------------------')

                    # print('\n------------------------cosine A---------------------------')
                    # print(adj_h_n)
                    # print('\n-----------------------------------------------------------')
                    adj_std, adj_mean = torch.std_mean(adj_h_n)
                    adj_h_n_ms = torch.exp(-torch.square((adj_h_n - adj_mean) / (adj_std + 0.000001)))
                    adj_mlp = self.transformA(adj_h_n_ms)
                    adj_act = self.activation(adj_mlp)

                    # normalisation
                    adj_act_std, adj_act_mean = torch.std_mean(adj_act)
                    adj_act_ms = torch.exp(-torch.square((adj_act - adj_act_mean) / adj_act_std))
                    adj_act_ms = adj_act_ms * diag
                    adj_mask = adj_act_ms > 0.1
                    adj_mask = adj_mask.long()
                    adj_act_ms = adj_act_ms * adj_mask

                    adj_sum = torch.sum(adj_act_ms, 1)
                    # adj_sum = adj_sum.expand(h_s.shape[0], h_s.shape[2], h_s.shape[2])
                    adj_result = adj_act_ms / (adj_sum + 0.000001)
                    adj = [adj_result, adj_result]
                    learned_adj = adj
                else:
                    h_s_t = h_s.permute((0, 2, 1))
                    h_s_t_n = h_s_t / (torch.norm(h_s_t, dim=(2), keepdim=True).expand(-1, -1, h_s_t.shape[2]) + 0.000001)
                    h_s_n = h_s / (torch.norm(h_s, dim=(1), keepdim=True).expand(-1, h_s.shape[1], h_s.shape[2]) + 0.000001)
                    adj_h_n = torch.bmm(h_s_t_n, h_s_n)
                    if (adj_h_n.dim() > 3):
                        adj_h_n = torch.squeeze(adj_h_n)
                    # print('-------------cosine A----------------')
                    adj_h_n_attention = adj_h_n.permute((1, 0, 2))
                    attn_output, attn_output_weights = self.attention(adj_h_n_attention, adj_h_n_attention,
                                                                      adj_h_n_attention)

                    adj_attention = torch.einsum('ncw,nwv->ncv', attn_output_weights, adj_h_n)
                    # adj_attention = torch.einsum('ncw,nwv->ncv', attn_output_weights, adj_h_n_norm)
                    adj_attention_mean = torch.mean(adj_attention, 0)
                    adj_h_n = adj_attention_mean
                    # print('-------------attention A----------------')
                    adj_std, adj_mean = torch.std_mean(adj_h_n)
                    adj_h_n_ms = torch.exp(-torch.square((adj_h_n - adj_mean) / (adj_std + 0.000001)))
                    adj_mlp = self.transformA(adj_h_n_ms)
                    adj_act = self.activation(adj_mlp)

                    # normalisation
                    # print('-------------MLP A----------------')
                    adj_act_std, adj_act_mean = torch.std_mean(adj_act)
                    adj_act_ms = torch.exp(-torch.square((adj_act - adj_act_mean) / adj_act_std + 0.000001))
                    adj_act_ms = adj_act_ms * diag
                    adj_mask = adj_act_ms > 0.1

                    adj_mask = adj_mask.long()
                    adj_act_ms = adj_act_ms * adj_mask
                    # print('-------------normlasization A----------------')
                    adj_sum = torch.sum(adj_act_ms, 1)
                    # adj_sum = adj_sum.expand(h_s.shape[0], h_s.shape[2], h_s.shape[2])
                    adj_result = adj_act_ms / (adj_sum + 0.000001)
                    adj = [adj_result, adj_result]
                    learned_adj = adj


            # retrieve maximum information from neighbors
            xs_hat, repr_s = self.spatial_decoder(x=x_s, m=m_s, h=h_s, u=u_s, adj=adj,
                                                    cached_support=cached_support)  # receive messages from neighbors (no self-loop!)
            # readout of imputation state + mask to retrieve imputations
            # prepare inputs
            x_s = torch.where(m_s, x_s, xs_hat)
            inputs = [x_s, m_s]
            if u_s is not None:
                inputs.append(u_s)
            inputs = torch.cat(inputs, dim=1)  # x_hat_2 + mask + exogenous
            # update state with original sequence filled using imputations
            h = self.update_state(inputs, h, adj)
            # store imputations and states
            states.append(torch.stack(h, dim=0))
            representations.append(repr_s)
            if step == steps - 1 :
                # torch.set_printoptions(threshold=10_000, linewidth=300)
                # print('learned graph: ')
                # print(adj_act_ms)
                h_enc = h[-1]

        # Aggregate outputs -> [batch, features, nodes, steps]
        states = torch.stack(states, dim=-1)
        representations = torch.stack(representations, dim=-1)

        return representations, states, learned_adj, h_enc


class Decoder(nn.Module):
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
                 layer_norm=False):
        super(Decoder, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.u_size = int(u_size) if u_size is not None else 0
        self.n_layers = int(n_layers)
        rnn_input_size = 2 * self.input_size + self.u_size  # input + mask + (eventually) exogenous

        # Spatio-temporal encoder (rnn_input_size -> hidden_size)
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(GCGRUCell(d_in=rnn_input_size if i == 0 else self.hidden_size,
                                        num_units=self.hidden_size, support_len=support_len, order=kernel_size))
            if layer_norm:
                self.norms.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size))
            else:
                self.norms.append(nn.Identity())
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # Fist stage readout
        self.first_stage = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.input_size, kernel_size=1)

        # Spatial decoder (rnn_input_size + hidden_size -> hidden_size)
        self.spatial_decoder = SpatialDecoder(d_in=rnn_input_size + self.hidden_size,
                                              d_model=self.hidden_size,
                                              d_out=self.input_size,
                                              support_len=2,
                                              order=decoder_order,
                                              attention_block=global_att)

        # Hidden state initialization embedding
        if n_nodes is not None:
            self.h0 = self.init_hidden_states(n_nodes)
        else:
            self.register_parameter('h0', None)

    def init_hidden_states(self, n_nodes):
        h0 = []
        for l in range(self.n_layers):
            std = 1. / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
            vals = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes))
            h0.append(nn.Parameter(vals))
        return nn.ParameterList(h0)

    def get_h0(self, x):
        if self.h0 is not None:
            return [h.expand(x.shape[0], -1, -1) for h in self.h0]
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device)] * self.n_layers

    def update_state(self, x, h, adj):
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer], adj))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h

    def forward(self, x, adj, h_enc, mask=None, u=None, h=None, cached_support=False):
        # x:[batch, features, nodes, steps]

        *_, steps = x.size()

        # infer all valid if mask is None
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)

        # init hidden state using node embedding or the empty state
        if h is None:
            h = self.get_h0(x)
            h[-1] = torch.mul(h[-1], h_enc)
        elif not isinstance(h, list):
            h = [*h]

        # Temporal conv
        predictions, imputations, states = [], [], []
        representations = []
        for step in range(steps):
            # x_s = x[..., step]
            # m_s = mask[..., step]
            # h_s = h[-1]
            # u_s = u[..., step] if u is not None else None
            #
            # # retrieve maximum information from neighbors
            # xs_hat, repr_s = self.spatial_decoder(x=x_s, m=m_s, h=h_s, u=u_s, adj=adj,
            #                                         cached_support=cached_support)  # receive messages from neighbors (no self-loop!)
            # # readout of imputation state + mask to retrieve imputations
            # # prepare inputs
            # x_s = torch.where(m_s, x_s, xs_hat)
            # inputs = [x_s, m_s]
            # if u_s is not None:
            #     inputs.append(u_s)
            # inputs = torch.cat(inputs, dim=1)  # x_hat_2 + mask + exogenous
            # # update state with original sequence filled using imputations
            # h = self.update_state(inputs, h, adj)
            # # store imputations and states
            # # firstly impute missing values with predictions from state
            # xs_hat_1 = self.first_stage(h_s)
            # # fill missing values in input with prediction
            # x_s = torch.where(m_s, x_s, xs_hat_1)
            #
            # imputations.append(xs_hat_1)
            # predictions.append(xs_hat)
            # states.append(torch.stack(h, dim=0))
            # representations.append(repr_s)
            x_s = x[..., step]
            m_s = mask[..., step]
            h_s = h[-1]
            u_s = u[..., step] if u is not None else None
            # firstly impute missing values with predictions from state
            xs_hat_1 = self.first_stage(h_s)
            # fill missing values in input with prediction
            x_s = torch.where(m_s, x_s, xs_hat_1)
            # prepare inputs
            # retrieve maximum information from neighbors
            xs_hat_2, repr_s = self.spatial_decoder(x=x_s, m=m_s, h=h_s, u=u_s, adj=adj,
                                                    cached_support=cached_support)  # receive messages from neighbors (no self-loop!)
            # readout of imputation state + mask to retrieve imputations
            # prepare inputs
            x_s = torch.where(m_s, x_s, xs_hat_2)
            inputs = [x_s, m_s]
            if u_s is not None:
                inputs.append(u_s)
            inputs = torch.cat(inputs, dim=1)  # x_hat_2 + mask + exogenous
            # update state with original sequence filled using imputations
            h = self.update_state(inputs, h, adj)
            # store imputations and states
            imputations.append(xs_hat_2)
            predictions.append(xs_hat_1)
            states.append(torch.stack(h, dim=0))
            representations.append(repr_s)


        # Aggregate outputs -> [batch, features, nodes, steps]
        imputations = torch.stack(imputations, dim=-1)
        predictions = torch.stack(predictions, dim=-1)
        states = torch.stack(states, dim=-1)
        representations = torch.stack(representations, dim=-1)

        return imputations, predictions, representations, states


class EncoderDecoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size,
                 ff_dropout,
                 n_layers=1,
                 dropout=0.,
                 n_nodes=None,
                 support_len=2,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 u_size=0,
                 embedding_size=0,
                 layer_norm=False,
                 merge='mlp'):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            dropout=dropout,
                            n_nodes=n_nodes,
                            support_len=support_len,
                            kernel_size=kernel_size,
                            decoder_order=decoder_order,
                            global_att=global_att,
                            u_size=u_size,
                            layer_norm=layer_norm)
        self.decoder = Decoder(input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            dropout=dropout,
                            n_nodes=n_nodes,
                            support_len=support_len,
                            kernel_size=kernel_size,
                            decoder_order=decoder_order,
                            global_att=global_att,
                            u_size=u_size,
                            layer_norm=layer_norm)

        if n_nodes is None:
            embedding_size = 0
        if embedding_size > 0:
            self.emb = nn.Parameter(torch.empty(embedding_size, n_nodes))
            nn.init.kaiming_normal_(self.emb, nonlinearity='relu')
        else:
            self.register_parameter('emb', None)

        if merge == 'mlp':
            self._impute_from_states = True
            # Update for DGLA
            self.out = nn.Sequential(
                nn.Conv2d(in_channels=2 * hidden_size + input_size + embedding_size,
                          out_channels=ff_size, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Conv2d(in_channels=ff_size, out_channels=input_size, kernel_size=1)
            )
        elif merge in ['mean', 'sum', 'min', 'max']:
            self._impute_from_states = False
            self.out = getattr(torch, merge)
        else:
            raise ValueError("Merge option %s not allowed." % merge)
        self.supp = None

    def forward(self, x, adj, mask=None, u=None, cached_support=False):
        if cached_support and (self.supp is not None):
            supp = self.supp
        else:
            supp = SpatialConvOrderK.compute_support(adj, x.device)
            self.supp = supp if cached_support else None
        # Encoder
        enc_repr, states, learned_adj, h_enc = self.encoder(x, supp, mask=mask, u=u, cached_support=cached_support)

        # Decoder
        fwd_out, fwd_pred, fwd_repr, _ = self.decoder(x, learned_adj, h_enc, mask=mask, u=u, cached_support=cached_support)

        if self._impute_from_states:
            inputs = [fwd_repr, mask]
            if self.emb is not None:
                b, *_, s = fwd_repr.shape  # fwd_h: [batches, channels, nodes, steps]
                inputs += [self.emb.view(1, *self.emb.shape, 1).expand(b, -1, -1, s)]  # stack emb for batches and steps
            imputation = torch.cat(inputs, dim=1)
            imputation = self.out(imputation)
        else:
            imputation = torch.stack([fwd_out], dim=1)
            imputation = self.out(imputation, dim=1)

        predictions = torch.stack([fwd_out,fwd_pred], dim=0)

        return imputation, predictions


class ClassifierEncoder(nn.Module):
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
        super(ClassifierEncoder, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.u_size = int(u_size) if u_size is not None else 0
        self.n_layers = int(n_layers)
        rnn_input_size = 2 * self.input_size + self.u_size  # input + mask + (eventually) exogenous

        # Spatio-temporal encoder (rnn_input_size -> hidden_size)
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(GCGRUCell(d_in=rnn_input_size if i == 0 else self.hidden_size,
                                        num_units=self.hidden_size, support_len=support_len, order=kernel_size))
            if layer_norm:
                self.norms.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size))
            else:
                self.norms.append(nn.Identity())
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # Spatial decoder (rnn_input_size + hidden_size -> hidden_size)
        self.spatial_decoder = SpatialDecoder(d_in=rnn_input_size + self.hidden_size,
                                              d_model=self.hidden_size,
                                              d_out=self.input_size,
                                              support_len=2,
                                              order=decoder_order,
                                              attention_block=global_att)

        # Hidden state initialization embedding
        if n_nodes is not None:
            self.h0 = self.init_hidden_states(n_nodes)
        else:
            self.register_parameter('h0', None)

        # learn a adj A
        # self.activation = nn.Sigmoid()
        self.activation = nn.LeakyReLU()
        # self.activation = nn.ReLU()
        self.transformA = nn.Linear(n_nodes, n_nodes, bias=True)
        self.attention = nn.MultiheadAttention(n_nodes, 1)
        self.adj_attention = nn.MultiheadAttention(embed_dim=16, num_heads=1)

        self.n_class = n_class
        # self.classifier = nn.Sequential(
        #     nn.Linear(n_nodes * self.hidden_size, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, self.n_class)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(n_nodes * self.hidden_size, n_nodes * self.hidden_size),
            nn.ReLU(),
            nn.Linear(n_nodes * self.hidden_size, self.n_class)
        )



    def init_hidden_states(self, n_nodes):
        h0 = []
        for l in range(self.n_layers):
            std = 1. / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
            vals = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes))
            xavier_norm = torch.nn.init.xavier_normal(torch.empty(self.hidden_size, n_nodes))
            kaiming_norm = torch.nn.init.kaiming_normal_(torch.empty(self.hidden_size, n_nodes), mode='fan_out')
            h0.append(nn.Parameter(vals))
            # h0.append(nn.Parameter(kaiming_norm))
        return nn.ParameterList(h0)

    def get_h0(self, x):
        if self.h0 is not None:
            observe = self.h0
            h0 = [h.expand(x.shape[0], -1, -1) for h in self.h0]
            return h0
           # return [h.expand(x.shape[0], -1, -1) for h in self.h0]
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device)] * self.n_layers

    def update_state(self, x, h, adj):
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer], adj))
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
            if(step != 0):
                if (attention_adj==True):

                    # attention
                    h_s_t = h_s.permute((2, 0, 1))
                    attn_output1, attn_output_weights1 = self.adj_attention(h_s_t, h_s_t, h_s_t)
                    adj_h_n = attn_output_weights1

                    # adj_h_n_sum = torch.sum(adj_h_n, 1)
                    # adj_h_n = adj_h_n / adj_h_n_sum
                    # attn_output, attn_output_weights = self.attention(adj_h_n_attention, adj_h_n_attention, adj_h_n_attention)
                    # adj_h_n_dia = adj_h_n.diagonal(dim1=1, dim2=2)
                    # adj_h_n_dia_unsq = torch.unsqueeze(adj_h_n_dia, dim=2)
                    # adj_h_n_dia_exp = adj_h_n_dia.expand(-1, -1, 36)
                    # adj_h_n_norm = adj_h_n / adj_h_n_dia_unsq
                    # adj_h_n_attention = adj_h_n_norm.permute((1, 0, 2))
                    adj_h_n_attention = adj_h_n.permute((1, 0, 2))
                    attn_output, attn_output_weights = self.attention(adj_h_n_attention, adj_h_n_attention, adj_h_n_attention)

                    adj_attention = torch.einsum('ncw,nwv->ncv', attn_output_weights, adj_h_n)
                    # adj_attention = torch.einsum('ncw,nwv->ncv', attn_output_weights, adj_h_n_norm)
                    adj_attention_mean = torch.mean(adj_attention, 0)
                    adj_h_n = adj_attention_mean
                    # print('\n--------------------Attention Mean A-----------------------')
                    # print(adj_attention_mean)
                    # print('\n-----------------------------------------------------------')

                    # print('\n------------------------cosine A---------------------------')
                    # print(adj_h_n)
                    # print('\n-----------------------------------------------------------')
                    adj_std, adj_mean = torch.std_mean(adj_h_n)
                    adj_h_n_ms = torch.exp(-torch.square((adj_h_n - adj_mean) / (adj_std + 0.000001)))
                    adj_mlp = self.transformA(adj_h_n_ms)
                    adj_act = self.activation(adj_mlp)

                    # normalisation
                    adj_act_std, adj_act_mean = torch.std_mean(adj_act)
                    adj_act_ms = torch.exp(-torch.square((adj_act - adj_act_mean) / adj_act_std))
                    adj_act_ms = adj_act_ms * diag
                    adj_mask = adj_act_ms > 0.1
                    adj_mask = adj_mask.long()
                    adj_act_ms = adj_act_ms * adj_mask

                    adj_sum = torch.sum(adj_act_ms, 1)
                    # adj_sum = adj_sum.expand(h_s.shape[0], h_s.shape[2], h_s.shape[2])
                    adj_result = adj_act_ms / (adj_sum + 0.000001)
                    adj = [adj_result, adj_result]
                    learned_adj = adj
                else:
                    h_s_t = h_s.permute((0, 2, 1))
                    h_s_t_n = h_s_t / (torch.norm(h_s_t, dim=(2), keepdim=True).expand(-1, -1, h_s_t.shape[2]) + 0.000001)
                    h_s_n = h_s / (torch.norm(h_s, dim=(1), keepdim=True).expand(-1, h_s.shape[1], h_s.shape[2]) + 0.000001)
                    adj_h_n = torch.bmm(h_s_t_n, h_s_n)
                    if (adj_h_n.dim() > 3):
                        adj_h_n = torch.squeeze(adj_h_n)
                    # print('-------------cosine A----------------')
                    adj_h_n_attention = adj_h_n.permute((1, 0, 2))
                    attn_output, attn_output_weights = self.attention(adj_h_n_attention, adj_h_n_attention,
                                                                      adj_h_n_attention)

                    adj_attention = torch.einsum('ncw,nwv->ncv', attn_output_weights, adj_h_n)
                    # adj_attention = torch.einsum('ncw,nwv->ncv', attn_output_weights, adj_h_n_norm)
                    adj_attention_mean = torch.mean(adj_attention, 0)
                    adj_h_n = adj_attention_mean
                    # print('-------------attention A----------------')
                    adj_std, adj_mean = torch.std_mean(adj_h_n)
                    adj_h_n_ms = torch.exp(-torch.square((adj_h_n - adj_mean) / (adj_std + 0.000001)))
                    adj_mlp = self.transformA(adj_h_n_ms)
                    adj_act = self.activation(adj_mlp)

                    # normalisation
                    # print('-------------MLP A----------------')
                    adj_act_std, adj_act_mean = torch.std_mean(adj_act)
                    adj_act_ms = torch.exp(-torch.square((adj_act - adj_act_mean) / (adj_act_std + 0.000001)))
                    adj_act_ms = adj_act_ms * diag
                    adj_mask = adj_act_ms > 0.1

                    adj_mask = adj_mask.long()
                    adj_act_ms = adj_act_ms * adj_mask
                    # print('-------------normlasization A----------------')
                    adj_sum = torch.sum(adj_act_ms, 1)
                    # adj_sum = adj_sum.expand(h_s.shape[0], h_s.shape[2], h_s.shape[2])
                    adj_result = adj_act_ms / (adj_sum + 0.000001)
                    adj = [adj_result, adj_result]
                    learned_adj = adj


            # retrieve maximum information from neighbors
            xs_hat, repr_s = self.spatial_decoder(x=x_s, m=m_s, h=h_s, u=u_s, adj=adj,
                                                    cached_support=cached_support)  # receive messages from neighbors (no self-loop!)
            # readout of imputation state + mask to retrieve imputations
            # prepare inputs
            x_s = torch.where(m_s, x_s, xs_hat)
            inputs = [x_s, m_s]
            if u_s is not None:
                inputs.append(u_s)
            inputs = torch.cat(inputs, dim=1)  # x_hat_2 + mask + exogenous
            # update state with original sequence filled using imputations
            h = self.update_state(inputs, h, adj)
            # store imputations and states
            states.append(torch.stack(h, dim=0))
            representations.append(repr_s)
            if step == steps - 1 :
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

        return representations, states, learned_adj, h_enc, output

