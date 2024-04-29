import torch
from einops import rearrange
from torch import nn

from ..layers.rits import BRITSC
from ...utils.parser_utils import str_to_bool


class BRITSCLASSIFIER(nn.Module):
    def __init__(self,
                 d_in,
                 ff_dropout,
                 d_dim,
                 d_hidden=64,
                 n_class=2):
        super(BRITSCLASSIFIER, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.britsc = BRITSC(input_size=self.d_in,
                              ff_dropout=ff_dropout,
                              hidden_size=self.d_hidden,
                              n_class=n_class)
        self.d_dim = d_dim
    def forward(self, x, mask=None, u=None, **kwargs):
        # x: [batches, steps, nodes, channels] -> [batches, channels, nodes, steps]
        # x = rearrange(x, 'b s n c -> b c n s')
        # x = x[:, :, :self.adj[0].shape[0], :]
        # mask = mask[:, :, :self.adj[0].shape[0], :]

        # if mask is not None:
        #     mask = rearrange(mask, 'b s n c -> b c n s')

        # imputation: [batches, channels, nodes, steps] prediction: [4, batches, channels, nodes, steps]

        # only focus on numerical information
        # x: [batches, steps, nodes, channels] -> [batches, channels, nodes, steps]
        x = rearrange(x, 'b s n c -> b c s n')
        x = x[:, :, :, :self.d_dim].squeeze()
        predictions, states, output = self.britsc(x, mask=mask)
        # In evaluation stage impute only missing values
        # if self.impute_only_holes and not self.training:
        #     imputation = torch.where(mask, x, imputation)
        # out: [batches, channels, nodes, steps] -> [batches, steps, nodes, channels]
        # imputation = torch.transpose(imputation, -3, -1)
        # prediction = torch.transpose(prediction, -3, -1)

        if self.training:
            return predictions, states, output
        return predictions, output

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-in', type=int, default=64)
        parser.add_argument('--d-hidden', type=int, default=64)
        parser.add_argument('--ff-dropout', type=int, default=0.)
        parser.add_argument('--d-dim', type=int, default=36)
        return parser