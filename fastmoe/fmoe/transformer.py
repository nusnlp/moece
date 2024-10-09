r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import math
import torch
import torch.nn as nn
from .layers import FMoE
from .linear import FMoELinear
from .fastermoe.config import switch_from_env


class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x


def gated_relu(x):
    return 0.5 * x * (1.0 + torch.tanh(
                                math.sqrt(2.0 / math.pi) * \
                                    (x + 0.044715 * torch.pow(x, 3.0))
                            ))


class T5DenseGatedActDenseExpert(nn.Module):
    def __init__(self, num_expert, d_model, d_hidden, activation, dropout_rate=0.1, gelu_layer=None, rank=0):
        super().__init__()
        if gelu_layer is None:
            print('New layer is created', flush=True)
            gelu_layer = FMoELinear(num_expert, d_model, d_hidden, bias=False, rank=0)
        else:
            print('Reusing gelu_layer module', flush=True)
        self.wi = nn.ModuleList([
                    gelu_layer,
                    FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
                  ])
        self.wo = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_fn = activation

    def forward(self, hidden_states, fwd_expert_count):
        hidden_gelu = self.act_fn(self.wi[0](hidden_states, fwd_expert_count))
        hidden_linear = self.wi[1](hidden_states, fwd_expert_count)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states, fwd_expert_count)
        return hidden_states



class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        d_dropout=0.1,
        expert_dp_comm="none",
        expert_type=None,
        share_gelu_layer=False,
        expert_rank=0,
        **kwargs
    ):
        def one_expert(d_model):
            if expert_type == 't5':
                return T5DenseGatedActDenseExpert(1, d_model, d_hidden, gated_relu, d_dropout, gelu_layer=gelu_layer, rank=0)
            else:
                return _Expert(1, d_model, d_hidden, activation, rank=0)
        
        if share_gelu_layer:
            gelu_layer = FMoELinear(1, d_model, d_hidden, bias=False, rank=0)
            # gelu_layer = FMoELinear(1, 1, d_hidden, bias=False, rank=0)
        else:
            gelu_layer = None
        expert = one_expert
        super().__init__(num_expert=num_expert, d_model=d_model, expert=expert, **kwargs)
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        return output.reshape(original_shape)
