# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Dict, List, NamedTuple, Optional

import torch
from torch import Tensor

from fairseq.models import FairseqEncoder
from fairseq.models.fairseq_encoder import EncoderOut

from .t5_moe_modules import T5_MOE_Stack


MOE_EncoderOut = NamedTuple(
    "MOE_EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
        ("moe_loss", Optional[Tensor]),  # B x moe_layer
        ("moe_gate_logits", Optional[Tensor]),  # B x moe_layer x dim
    ],
)


class T5_MOE_Encoder(FairseqEncoder):
    """
    T5 MoE encoder for T5 model.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary=dictionary)

        # self.padding_idx = embed_tokens.padding_idx
        self.padding_idx = 0

        encoder_config = copy.deepcopy(args)
        encoder_config.is_decoder = False
        if hasattr(encoder_config, "moe_location") and "encoder" not in encoder_config.moe_location:
            encoder_config.moe_freq = 0
        self.max_source_positions = 512

        self.t5_stack = T5_MOE_Stack(config=encoder_config, embed_tokens=embed_tokens)

    def forward(self, src_tokens, src_lengths, cls_input=None, return_all_hiddens=False, **unused):
        encoder_padding_mask = ~ src_tokens.eq(self.padding_idx)
        # print('[E]:', src_tokens, flush=True)
        # print('[E_]:', encoder_padding_mask, flush=True)
        stack_output = self.t5_stack.forward(src_tokens, encoder_padding_mask)
        hidden_states = stack_output[0].transpose(0, 1)
        # print('>>> hs:', hidden_states.shape, flush=True)
        if self.training:
            # print('>>> MOE encoder out', flush=True)
            moe_loss = stack_output[-2]
            moe_gate_logits = stack_output[-1]
            encoder_out = MOE_EncoderOut(
                encoder_out=hidden_states,
                encoder_padding_mask=encoder_padding_mask,
                encoder_embedding=None,
                encoder_states=None,
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                moe_loss=moe_loss,
                moe_gate_logits=moe_gate_logits,
            )
        else:
            # print('>>> normal encoder out', flush=True)
            encoder_out = EncoderOut(
                encoder_out=hidden_states,
                encoder_padding_mask=encoder_padding_mask,
                encoder_embedding=None,
                encoder_states=None,
                src_tokens=src_tokens,
                src_lengths=src_lengths,
            )
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions

    # def upgrade_state_dict_named(self, state_dict, name):
    #     """Upgrade a (possibly old) state dict for new versions of fairseq."""
    #     model_dict = self.state_dict()
    #     pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    #     # print("\n==== before encoder:\n", model_dict) 
    #     # model_dict.update(pretrained_dict)
    #     # print("\n==== after encoder:\n", model_dict) 
    #     return model_dict

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        # print('>>> reorder encoder', flush=True)
        new_encoder_out: Dict[str, Tensor] = {}

        new_encoder_out["encoder_out"] = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_out["encoder_padding_mask"] = (
            encoder_out.encoder_padding_mask
            if encoder_out.encoder_padding_mask is None
            else encoder_out.encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_out["encoder_embedding"] = (
            encoder_out.encoder_embedding
            if encoder_out.encoder_embedding is None
            else encoder_out.encoder_embedding.index_select(0, new_order)
        )

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)
        
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out["encoder_out"],  # T x B x C
            encoder_padding_mask=new_encoder_out["encoder_padding_mask"],  # B x T
            encoder_embedding=new_encoder_out["encoder_embedding"],  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )
