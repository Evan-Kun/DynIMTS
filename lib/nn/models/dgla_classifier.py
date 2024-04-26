import torch
from einops import rearrange
from torch import nn

from ..layers import ClassifierEncoder
from ...utils.parser_utils import str_to_bool

class DGLACLASSIFIER(nn.Module):
    def __init__(self,
                 adj,
                 d_in,
                 d_hidden,
                 ff_dropout,
                 n_layers=1,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 d_u=0,
                 d_emb=0,
                 layer_norm=False,
                 n_class=2):
        super(DGLACLASSIFIER, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_u = int(d_u) if d_u is not None else 0
        self.d_emb = int(d_emb) if d_emb is not None else 0
        self.register_buffer('adj', torch.tensor(adj).float())
        self.dglac = ClassifierEncoder(input_size=self.d_in,
                             hidden_size=self.d_hidden,
                             n_nodes=self.adj.shape[0],
                             n_layers=n_layers,
                             dropout=ff_dropout,
                             kernel_size=kernel_size,
                             decoder_order=decoder_order,
                             global_att=global_att,
                             u_size=self.d_u,
                             layer_norm=layer_norm,
                             n_class=n_class
        )


    def forward(self, x, mask=None, u=None, **kwargs):
        # x: [batches, steps, nodes, channels] -> [batches, channels, nodes, steps]
        x = rearrange(x, 'b s n c  -> b c n s')
        if mask is not None:
            mask = rearrange(mask, 'b s n c -> b c n s')

        if u is not None:
            u = rearrange(u, 'b s n c -> b c n s')

        representation, states, learned_adj, h_enc, output = self.dglac(x, self.adj, mask=mask, u=u, cached_support=self.training, )
        predictions = representation
        if self.training:
            return predictions, states, output
        return predictions, output

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-hidden', type=int, default=64)
        # parser.add_argument('--d-f', type=int, default=64)
        parser.add_argument('--ff-dropout', type=int, default=0.)
        parser.add_argument('--n-layers', type=int, default=1)
        parser.add_argument('--kernel-size', type=int, default=2)
        parser.add_argument('--decoder-order', type=int, default=1)
        parser.add_argument('--d-u', type=int, default=0)
        parser.add_argument('--d-emb', type=int, default=8)
        parser.add_argument('--layer-norm', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--global-att', type=str_to_bool, nargs='?', const=True, default=False)
        # parser.add_argument('--merge', type=str, default='mlp')
        return parser