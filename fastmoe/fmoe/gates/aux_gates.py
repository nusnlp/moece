r"""
Balanced gate with Switch Transformer's policy (Google, 2021)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_gate import BaseGate
from .utils import limit_by_capacity
import fmoe_cuda as fmoe_native


class MultiLayerGate(nn.Module):
    def __init__(self, d_in, gate_hidden_dims, d_out, gate_bias=True, class_dim=None):
        super().__init__()
        gate_hidden_dims = (gate_hidden_dims or []) + [d_out]
        gate_layers = []
        last_dim = d_in
        for hidden_dim in gate_hidden_dims:
            gate_layers.append(nn.Linear(last_dim, hidden_dim, bias = gate_bias))
            last_dim = hidden_dim
        self.gate_layers = nn.ModuleList(gate_layers)
        if class_dim is not None:
            assert len(gate_hidden_dims) > 1, "Please specify the --gate-hidden-dims"
            in_dim = gate_hidden_dims[-2] # the last one is num of expert
            self.class_head = nn.Linear(in_dim, class_dim)

    def forward(self, x):
        last_hidden_output = None
        second_to_last_idx = len(self.gate_layers) - 2
        for i, gate_layer in enumerate(self.gate_layers):
            x = gate_layer(x)
            if i == second_to_last_idx:
                last_hidden_output = x
        if hasattr(self, 'class_head'):
            gate_logit = self.class_head(last_hidden_output)
        else:
            gate_logit = last_hidden_output
        return x, gate_logit


class InternalAuxNaiveGate(BaseGate):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2, gate_bias=True, gate_hidden_dims=None, class_dim=None, **kwargs):
        super().__init__(num_expert, world_size)
        self.gate = MultiLayerGate(d_model, gate_hidden_dims, self.tot_expert, gate_bias, class_dim=class_dim)
        self.top_k = top_k

    def forward(self, inp, return_all_scores=False):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        gate, gate_logits = self.gate(inp)
        self.set_logits(gate_logits)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        # dummy loss
        self.set_loss(torch.zeros(1, requires_grad=True).cuda())

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate, gate_logits
        return gate_top_k_idx, gate_score, gate_logits
    
    def set_logits(self, logits):
        self.logits = logits

    def get_logits(self, clear=True):
        logits = self.logits
        if clear:
            self.logits = None
        return logits


class AuxNaiveGate(InternalAuxNaiveGate):
    def __init__(self, d_model, num_expert, world_size,
            top_k=2, capacity=(1.2, 2.4), random_routing=True, gate_bias=True, gate_hidden_dims=None, class_dim=None, **kwargs):
        super().__init__(d_model, num_expert, world_size, top_k=top_k, gate_bias=gate_bias, gate_hidden_dims=gate_hidden_dims, class_dim=class_dim)
        self.capacity = capacity
        self.random_routing = random_routing

    def forward(self, x):
        naive_outs = super().forward(x, return_all_scores=True)
        topk_idx, topk_val, gate_score, gate_logits = naive_outs

        S = gate_score.shape[0]
        top1_idx = topk_idx.view((-1, self.top_k))[:, 0]
        c_e = torch.scatter_add(
                torch.zeros(self.tot_expert, device=top1_idx.device),
                0,
                top1_idx,
                torch.ones_like(top1_idx, dtype=torch.float),
                ) / S
        m_e = torch.mean(F.softmax(gate_score, dim=1), dim=0)
        loss = torch.mean(c_e * m_e) * (self.num_expert ** 2)
        self.set_loss(loss)

        return topk_idx, topk_val


class AuxSwitchGate(InternalAuxNaiveGate):
    r"""
    A switch gate implementation
    """

    def __init__(self, d_model, num_expert, world_size, top_k=1,
            switch_eps=.1, capacity=(1.2, 2.4), gate_bias=True, gate_hidden_dims=None, class_dim=None, **kwargs):
        assert top_k == 1, 'top_k should be 1 in switch'
        super().__init__(d_model, num_expert, world_size, top_k=1, gate_bias=gate_bias, gate_hidden_dims=gate_hidden_dims, class_dim=class_dim)
        self.switch_eps = switch_eps
        self.capacity = capacity

    def forward(self, inp):
        r"""
        The switch firstly conduct softmax and then calculates the top-1
        """
        score, gate_logits = self.gate(inp)
        self.set_logits(gate_logits)

        if self.training:
            # random uniform number from [1-eps, 1+eps]
            noise = torch.rand_like(score)
            noise = noise * 2 * self.switch_eps + 1.0 - self.switch_eps
            score += noise

        # fp32 softmax for numerical stability
        score = F.softmax(score.float(), dim=-1)

        top1_score, top1_idx = torch.topk(
            score, k=1, dim=-1, largest=True
        )  # [.. x top_k]
        top1_score = top1_score.to(dtype=inp.dtype)

        cap_rate = self.capacity[0 if self.training else 1]
        capacity = math.ceil(cap_rate * inp.shape[0] / self.num_expert)
        _new_lec, _new_gec, top1_idx = limit_by_capacity(
                top1_idx, self.num_expert, self.world_size, capacity)

        valid_idx = top1_idx[top1_idx > -1]
        fraction_expert = torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            ) / valid_idx.numel()
        prob_expert = score.sum(dim=0) / valid_idx.numel()
        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.set_loss(loss)
        return top1_idx, top1_score 


class AuxSwitchGate2(InternalAuxNaiveGate):
    r"""
    A switch gate implementation
    """

    def __init__(self, d_model, num_expert, world_size, top_k=1,
            switch_eps=.1, capacity=(1.2, 2.4), gate_bias=True, gate_hidden_dims=None, class_dim=None, **kwargs):
        assert top_k == 1, 'top_k should be 1 in switch'
        super().__init__(d_model, num_expert, world_size, top_k=1, gate_bias=gate_bias, gate_hidden_dims=gate_hidden_dims, class_dim=class_dim)
        self.switch_eps = switch_eps
        self.capacity = capacity

    def forward(self, inp):
        r"""
        The switch firstly conduct softmax and then calculates the top-1
        """
        score, gate_logits = self.gate(inp)
        self.set_logits(gate_logits)

        if self.training:
            # random uniform number from [1-eps, 1+eps]
            noise = torch.rand_like(score)
            noise = noise * 2 * self.switch_eps + 1.0 - self.switch_eps
            score += noise

        top1_score, top1_idx = torch.topk(
            score, k=1, dim=-1, largest=True
        )  # [.. x top_k]
        top1_score = top1_score

        # fp32 softmax for numerical stability
        score = F.softmax(score.float(), dim=-1).to(dtype=inp.dtype)
        top1_score = F.softmax(top1_score.float(), dim=-1).to(dtype=inp.dtype)

        cap_rate = self.capacity[0 if self.training else 1]
        capacity = math.ceil(cap_rate * inp.shape[0] / self.num_expert)
        _new_lec, _new_gec, top1_idx = limit_by_capacity(
                top1_idx, self.num_expert, self.world_size, capacity)

        valid_idx = top1_idx[top1_idx > -1]
        fraction_expert = torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            ) / valid_idx.numel()
        prob_expert = score.sum(dim=0) / valid_idx.numel()
        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.set_loss(loss)
        return top1_idx, top1_score 


class AuxGShardGate(InternalAuxNaiveGate):
    def __init__(self, d_model, num_expert, world_size,
            top_k=2, capacity=(1.2, 2.4), random_routing=True, gate_bias=True, gate_hidden_dims=None, class_dim=None, **kwargs):
        assert top_k == 2, 'topk should be 2 in gshard'
        super().__init__(d_model, num_expert, world_size, top_k=2, gate_bias=gate_bias, gate_hidden_dims=gate_hidden_dims, class_dim=class_dim)
        self.capacity = capacity
        self.random_routing = random_routing

    def forward(self, x):
        naive_outs = super().forward(x, return_all_scores=True)
        topk_idx, topk_val, gate_score, gate_logits = naive_outs

        S = gate_score.shape[0]
        top1_idx = topk_idx.view((-1, self.top_k))[:, 0]
        c_e = torch.scatter_add(
                torch.zeros(self.tot_expert, device=top1_idx.device),
                0,
                top1_idx,
                torch.ones_like(top1_idx, dtype=torch.float),
                ) / S
        m_e = torch.mean(F.softmax(gate_score, dim=1), dim=0)
        loss = torch.mean(c_e * m_e) * (self.num_expert ** 2)
        self.set_loss(loss)

        cap_rate = self.capacity[0 if self.training else 1]
        capacity = math.ceil(cap_rate * x.shape[0])
        capacity = capacity * self.top_k // (self.world_size * self.num_expert)
        capacity = torch.ones(self.num_expert * self.world_size,
                dtype=torch.int32, device=topk_idx.device) * capacity
        topk_idx = fmoe_native.prune_gate_by_capacity(topk_idx, capacity,
                self.num_expert, self.world_size)

        if self.training and self.random_routing:
            rand_routing_prob = torch.rand(gate_score.size(0), device=x.device)
            mask = (2 * topk_val[:, 1] < rand_routing_prob)
            topk_idx[:, 1].masked_fill_(mask, -1)

        return topk_idx, topk_val
