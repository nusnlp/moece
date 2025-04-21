import copy
import json
from typing import Optional, Tuple, Union

from beartype import beartype

import torch
import torch.nn.functional as F
from torch import nn

from ..t5.t5_modules import (
    T5LayerSelfAttention,
    T5LayerCrossAttention,
    T5LayerFF,
    T5DenseReluDense,
    T5DenseGatedActDense,
    T5LayerNorm,
)

from ..st_moe_pytorch.st_moe_pytorch import MoE, TopNGating
from ..st_moe_pytorch.soft_moe import SoftMoE
from fmoe import FMoETransformerMLP
from fmoe.gates import NaiveGate, NoisyGate, GShardGate, DCGate, SwipeGate, SwitchGate, AuxNaiveGate, AuxSwitchGate, AuxSwitchGate2, AuxGShardGate


def gate_producer(config):
    if config.moe_type.lower() == 'naive':
        gate = NaiveGate
    elif config.moe_type.lower() == 'noisy':
        gate = NoisyGate
    elif config.moe_type.lower() == 'gshard':
        gate = GShardGate
    elif config.moe_type.lower() == 'dcgate':
        gate = DCGate
    elif config.moe_type.lower() == 'swipe':
        gate = SwipeGate
    elif config.moe_type.lower() == 'switch':
        gate = SwitchGate
    elif config.moe_type.lower() == 'aux_naive':
        gate = AuxNaiveGate
    elif config.moe_type.lower() == 'aux_switch':
        gate = AuxSwitchGate
    elif config.moe_type.lower() == 'aux_switch2':
        gate = AuxSwitchGate2
    elif config.moe_type.lower() == 'aux_gshard':
        gate = AuxGShardGate
    else:
        raise ValueError("gate type {} is not recognized".format(config.moe_type))

    gate_args = {}
    gate_hidden_dims = json.loads(config.gate_hidden_dims) \
                        if hasattr(config, "gate_hidden_dims") and config.gate_hidden_dims \
                        else None
    if gate_hidden_dims is not None:
        gate_args['gate_hidden_dims'] = gate_hidden_dims
    gate_capacity = tuple(json.loads(config.gate_capacity)) \
                        if hasattr(config, "gate_capacity") and config.gate_capacity \
                        else None
    if gate_capacity is not None:
        gate_args['capacity'] = gate_capacity
    
    if config.gate_class_dim is not None and \
        config.moe_type.lower().startswith('aux'):
        gate_args['class_dim'] = config.gate_class_dim
    
    return gate, gate_args


# class T5_MOE_FF(nn.Module):
#     def __init__(self, config, is_moe=False, gate=None, **kwargs):
#         super().__init__()
#         if config.activation_fn.startswith('gated'):
#             self.ff_layer = T5DenseGatedActDense
#         else:
#             self.ff_layer = T5DenseReluDense
#         self.DenseReluDense = self.ff_layer(config)
#         self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
#         self.dropout = nn.Dropout(config.dropout_rate)
#         if is_moe:
#             expert_config = copy.deepcopy(config)
#             if hasattr(config, "expert_dropout"):
#                 expert_config.dropout_rate = config.expert_dropout
#             # the pre-trained FF becomes the first expert
#             experts = [self.DenseReluDense] + [self.ff_layer(expert_config) for _ in range(config.num_experts - 1)]
#             if config.moe_type == "hard":
#                 self.moe = MoE(
#                     dim=config.gate_dim,
#                     num_experts=config.num_experts,
#                     gating_top_n=config.gate_top_n,
#                     differentiable_topk=config.differentiable_topk,
#                     experts=experts,
#                     gate=gate) # if None, will create it's own gate
#             elif config.moe_type == "soft":
#                 self.moe = SoftMoE(
#                     dim=config.gate_dim,
#                     num_experts=config.num_experts,
#                     seq_len=config.n_positions // 2,
#                     experts=experts,
#                     offload_unused_experts_to_cpu=False,
#                 )
#             else:
#                 raise NotImplemented("Moe type {} is not found.".format(config.moe_type))
#             if hasattr(config, "gate_logits") and config.gate_logits:
#                 assert hasattr(self.moe.gate, 'get_logits') and callable(self.moe.gate.get_logits), \
#                     "Only gates with get_logits can be used with --gate-logit argument!"
#         else:
#             self.moe = None

#     def forward(self, hidden_states):
#         norm_x = self.layer_norm(hidden_states)
        
#         if self.moe is not None:
#             moe_out = self.moe(norm_x)
#             y = moe_out.outputs
#             # print('>>>> moe block hidden_states:', hidden_states.shape, flush=True)
#             moe_loss = moe_out.total_aux_loss
#         else:
#             y = self.DenseReluDense(norm_x)
#             # print('>>>> block hidden_states:', hidden_states.shape, flush=True)
#             moe_loss = None
#         layer_output = hidden_states + self.dropout(y)
#         return layer_output, moe_loss


class T5LayerFF_FMOE(nn.Module):
    def __init__(self, config, is_moe=False, gate=None):
        super().__init__()
        if config.activation_fn.startswith('gated'):
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseReluDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        if is_moe:
            # Make the MoE behaves similar like LoRA
            expert_dropout = getattr(config, "expert_dropout", config.dropout_rate)
            activation = nn.Sequential(
                nn.GELU(), # need to use approximate='tanh' to mimic T5, but not available on torch v1.1
                nn.Dropout(expert_dropout)
            )
            # if share_gate is True, the value doesn't matter
            gate_type, gate_args = gate_producer(config)
            expert_hidden = getattr(config, "expert_hidden", config.d_ff)
            self.moe = FMoETransformerMLP(
                        config.num_experts,
                        config.d_model,
                        expert_hidden,
                        activation=activation,
                        top_k=config.gate_top_n,
                        gate=gate_type,
                        gate_args=gate_args,
                    )
            if gate is not None:
                # TODO: test whether it works well (including the loss calc)
                self.moe.gate = gate
            
            if hasattr(config, "gate_logits") and config.gate_logits:
                assert hasattr(self.moe.gate, 'get_logits') and callable(self.moe.gate.get_logits), \
                    "Only gates with get_logits can be used with --gate-logit argument!"
            self.moe_scaling = getattr(config, "moe_scaling", 1)
            if config.freeze_backbone:
                for param in self.DenseReluDense.parameters():
                    param.requires_grad = False

    def forward(self, hidden_states):
        norm_x = self.layer_norm(hidden_states)
        y = self.DenseReluDense(norm_x)
        moe_loss, moe_logits = None, None
        if hasattr(self, "moe"):
            y = y + (self.moe_scaling * self.moe(norm_x))
            moe_loss = self.moe.gate.get_loss()
            if hasattr(self.moe.gate, 'get_logits'):
                moe_logits = self.moe.gate.get_logits()
            else:
                moe_logits = None
        layer_output = hidden_states + self.dropout(y)
        return layer_output, moe_loss, moe_logits


class T5LayerFF_Combined(nn.Module):
    def __init__(self, config, is_moe=False, gate=None):
        super().__init__()
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        if config.activation_fn.startswith('gated'):
            layer_class = T5DenseGatedActDense
        else:
            layer_class = T5DenseReluDense
        if is_moe:
            # Make the MoE behaves similar like LoRA
            expert_dropout = getattr(config, "expert_dropout", config.dropout_rate)
            # expert_config = copy.deepcopy(config)
            # expert_config.dropout_rate = expert_dropout
            # if share_gate is True, the value doesn't matter
            gate_type, gate_args = gate_producer(config)
            share_gelu_layer = getattr(config, "share_expert_gelu", False)
            expert_hidden = getattr(config, "expert_hidden", config.d_ff)
            self.moe = FMoETransformerMLP(
                        config.num_experts,
                        config.d_model,
                        expert_hidden,
                        d_dropout=expert_dropout,
                        top_k=config.gate_top_n,
                        gate=gate_type,
                        gate_args=gate_args,
                        expert_type='t5',
                        share_gelu_layer=share_gelu_layer,
                    )
            if gate is not None:
                # TODO: test whether it works well (including the loss calc)
                self.moe.gate = gate
            
            if hasattr(config, "gate_logits") and config.gate_logits:
                assert hasattr(self.moe.gate, 'get_logits') and callable(self.moe.gate.get_logits), \
                    "Only gates with get_logits can be used with --gate-logit argument!"
            self.moe_scaling = getattr(config, "moe_scaling", 1)
        else:
            self.DenseReluDense = layer_class(config)
        if config.freeze_non_MoE:
            layer_list = [self.layer_norm]
            if not is_moe:
                layer_list.append(self.DenseReluDense)
            for param_group in layer_list:
                for param in param_group.parameters():
                    param.requires_grad = False

    def forward(self, hidden_states):
        norm_x = self.layer_norm(hidden_states)
        moe_loss, moe_logits = None, None
        if hasattr(self, "moe"):
            y = self.moe_scaling * self.moe(norm_x)
            moe_loss = self.moe.gate.get_loss()
            if hasattr(self.moe.gate, 'get_logits'):
                moe_logits = self.moe.gate.get_logits()
            else:
                moe_logits = None
        else:
            y = self.DenseReluDense(norm_x)
        layer_output = hidden_states + self.dropout(y)
        return layer_output, moe_loss, moe_logits


class T5_MOE_Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, is_moe=False, gate=None):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config, has_relative_attention_bias=has_relative_attention_bias))

        self.is_moe = is_moe
        
        moe_layer = T5LayerFF_Combined if getattr(config, "merge_backbone", False) \
                        else T5LayerFF_FMOE
        self.layer.append(moe_layer(config, is_moe=is_moe, gate=gate))
        if config.freeze_non_MoE:
            layer_list = list(self.layer[:-1])
            for param_group in layer_list:
                for param in param_group.parameters():
                    param.requires_grad = False
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        head_mask=None,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states, attention_mask=attention_mask, position_bias=position_bias, head_mask=head_mask
        )
        hidden_states = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # Keep self-attention outputs and relative position weights

        if self.is_decoder:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                head_mask=head_mask,
            )
            hidden_states = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:]
            )  # Keep cross-attention outputs and relative position weights
        
        layer_output = self.layer[-1](hidden_states)
        hidden_states = layer_output[0]
        moe_loss = layer_output[1]
        moe_logits = layer_output[2]

        outputs = (hidden_states,) + outputs + (moe_loss, moe_logits) # add attentions if we output them
        
        # outputs = (hidden_states,) + outputs # add attentions if we output them
        # if self.training:
        #     outputs = outputs + (moe_loss,)
        return outputs  # hidden-states, (self-attention weights), (self-attention position bias), (cross-attention weights), \
        # (cross-attention position bias)


class T5_MOE_Logits(nn.Module):

    @beartype
    def __init__(
        self,
        moe: MoE,
        *,
        ff_before = None,
        ff_after = None
    ):
        super().__init__()
        dim = moe.dim

        self.moe = moe
        self.moe_prenorm = T5LayerNorm(dim)

        self.ff_before = ff_before
        self.ff_after = ff_after

    def forward(
        self,
        x,
        noise_gates = False,
        noise_mult = 1.
    ):

        # feedforward before
        if self.ff_before is not None:
            x = self.ff_before(x) + x

        # mixture of experts layer

        residual = x
        x, total_moe_loss, balance_loss, router_z_loss = self.moe(self.moe_prenorm(x), noise_gates = noise_gates, noise_mult = noise_mult)

        # feedforward after
        if self.ff_after is not None:
            x = x + residual
            x = self.ff_after(x)

        return x, total_moe_loss


class T5_MOE_Stack(nn.Module):
    def __init__(self, config, embed_tokens=None):
        super().__init__()
        self.config = config
        self.output_attentions = getattr(config, 'output_attentions', False)
        self.output_hidden_states = getattr(config, 'output_hidden_states', False)

        self.is_decoder = config.is_decoder
        self.embed_tokens = embed_tokens
        gate = None
        if hasattr(config, "share_gate") and config.share_gate:
            if config.moe_type == "hard":
                # The hard-coded parameters follow the default values in st_moe_pytorch.py
                gate = TopNGating(
                    config.gate_dim,
                    top_n = config.gate_top_n,
                    num_gates = config.num_experts,
                    straight_through_dispatch_tensor = True,
                    differentiable_topk = config.differentiable_topk,
                    differentiable_topk_fused= config.differentiable_topk,
                    threshold_train = 0.2,
                    threshold_eval = 0.2,
                    capacity_factor_train = 1.25,
                    capacity_factor_eval = 2.,
                )
            elif config.moe_type == 'soft':
                raise NotImplementedError("Soft MoE with shared gate has not been implemented")
            else:
                gate_type, gate_args = gate_producer(config)
                gate = gate_type(
                        config.d_model,
                        config.num_experts,
                        world_size=1,
                        top_k=config.gate_top_n,
                        **gate_args,
                        )
                # if gate_hidden_dims is not None:
                #     gate = gate_type(
                #             config.d_model,
                #             config.num_experts,
                #             world_size=1,
                #             top_k=config.gate_top_n,
                #             gate_hidden_dims=gate_hidden_dims,
                #             )
                # else:
                #     gate = gate_type(
                #             config.d_model,
                #             config.num_experts,
                #             world_size=1,
                #             top_k=config.gate_top_n,
                #             )
        if self.is_decoder:
            if config.tie_embeddings:
                self.logits = embed_tokens
            else:
                self.logits = nn.Linear(config.d_model, config.vocab_size, bias=False)
        layers = [T5_MOE_Block(config, has_relative_attention_bias=True)]
        for i in range(1, config.num_layers):
            is_moe_layer = config.moe_freq > 0 and ((i+1) % config.moe_freq == 0)
            layers.append(T5_MOE_Block(config, False, is_moe=is_moe_layer, gate=gate))

        self.block = nn.ModuleList(layers)
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        if config.freeze_non_MoE:
            layer_list = [self.embed_tokens, self.final_layer_norm]
            if self.is_decoder:
                layer_list.append(self.logits)
            for param_group in layer_list:
                for param in param_group.parameters():
                    param.requires_grad = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.logits

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length).to(inputs_embeds.device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                seq_ids = torch.arange(seq_length, device=inputs_embeds.device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(attention_mask)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -1e9 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
        # extended_attention_mask = (extended_attention_mask == extended_attention_mask.transpose(-1, -2))

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        if self.is_decoder:
            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

            # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
            # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
            # encoder_extended_attention_mask = (encoder_extended_attention_mask == encoder_extended_attention_mask.transpose(-1, -2))

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_layers

        all_hidden_states = ()
        all_attentions = ()
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        moe_outputs = []
        all_moe_logits = []
        for i, layer_module in enumerate(self.block):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i],
            )
            # layer_outputs is a tuple with:
            # hidden-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias), moe_outputs
            
            # if isinstance(layer_outputs[0], MixtureOfExpertsReturn):
            #     moe_outputs.append(layer_outputs.total_moe_loss)
            #     layer_outputs = layer_outputs.outputs
            if layer_module.is_moe:
                moe_loss = layer_outputs[-2]
                assert moe_loss is not None, "MoE layer shouldn't have a None loss"
                moe_outputs.append(moe_loss)
                if self.config.gate_logits:
                    moe_logits = layer_outputs[-1]
                    assert moe_logits is not None, "MoE layer shouldn't have a None gate logits"
                    all_moe_logits.append(moe_logits)
            layer_outputs = layer_outputs[:-2] # remove moe loss and logits
            
            # print('>>>> stack layer_outputs:', len(layer_outputs), flush=True)
            hidden_states = layer_outputs[0]
            # print('>>>> stack hidden_states:', hidden_states.shape, flush=True)
            
            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[2 if self.output_attentions else 1]
                if self.is_decoder:
                    encoder_decoder_position_bias = layer_outputs[4 if self.output_attentions else 2]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)  # We keep only self-attention weights for now

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        if self.training:
            outputs = outputs + (moe_outputs, all_moe_logits)
        # print('>>>> outputs', outputs, flush=True)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
