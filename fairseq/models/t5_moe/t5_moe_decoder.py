import copy
import logging

import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import FairseqIncrementalDecoder
from .t5_moe_modules import T5_MOE_Stack, T5_MOE_Logits
from ..st_moe_pytorch.st_moe_pytorch import MoE

logger = logging.getLogger(__name__)


class T5_MOE_Decoder(FairseqIncrementalDecoder):
    """T5 MoE Decoder to T5 model."""

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)

        decoder_config = copy.deepcopy(args)
        decoder_config.is_decoder = True
        if hasattr(decoder_config, "moe_location") and "decoder" not in decoder_config.moe_location:
            decoder_config.moe_freq = 0
        self.config = decoder_config
        self.t5_stack = T5_MOE_Stack(decoder_config, embed_tokens)
        if hasattr(decoder_config, "moe_location") and "logits" in decoder_config.moe_location:
            if "logits1" in decoder_config.moe_location:
                logit_experts = [self.t5_stack.logits] + \
                                    [nn.Linear(decoder_config.d_model, decoder_config.vocab_size, bias=False) \
                                        for _ in range(decoder_config.num_experts - 1)]
                ff_after = None
            elif "logits2" in decoder_config.moe_location:
                logit_experts = [nn.Linear(decoder_config.d_model, decoder_config.d_model, bias=False) \
                                    for _ in range(decoder_config.num_experts)]
                ff_after = self.t5_stack.logits
            else:
                raise ValueError("Please specify logits1 or logits2")
            moe = MoE(
                    dim=decoder_config.gate_dim,
                    num_experts=decoder_config.num_experts,
                    gating_top_n=decoder_config.gate_top_n,
                    differentiable_topk=decoder_config.differentiable_topk,
                    experts=logit_experts,
                  )
            self.moe_logit = T5_MOE_Logits(moe=moe, ff_after=ff_after)

        self.embed_tokens = embed_tokens
        self.tie_embeddings = args.tie_embeddings

    # def upgrade_state_dict_named(self, state_dict, name):
    #     """Upgrade a (possibly old) state dict for new versions of fairseq."""
    #     model_dict = self.state_dict()
    #     pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    #     model_dict.update(pretrained_dict) 
    #     return model_dict
    
    def forward(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args
    ):
        # print('[D]:', prev_output_tokens, flush=True)
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            **extra_args
        )
        if not features_only:
            if self.tie_embeddings:
                x = x * (self.config.d_model ** -0.5)
            logits_output = self.output_layer(x)
            if hasattr(self, "moe_logit"):
                x = logits_output[0]
                if extra is not None:
                    extra["moe_loss"].append(logits_output[-1])
            else:
                x = logits_output
        return x, extra

    def output_layer(self, features):
        if hasattr(self, "moe_logit"):
            return self.moe_logit(features)
        else:
            """Project features to the vocabulary size."""
            # project back to size of vocabulary
            return F.linear(features, self.t5_stack.get_output_embeddings().weight)

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        full_context_alignment=False,
        **unused,
    ):
        decoder_attention_mask = ~ prev_output_tokens.eq(0)
        decoder_attention_mask[:, 0] = True
        stack_output = self.t5_stack.forward(
            input_ids=prev_output_tokens,
            attention_mask=decoder_attention_mask,
            inputs_embeds=None,
            encoder_hidden_states=encoder_out.encoder_out.transpose(0, 1),
            encoder_attention_mask=encoder_out.encoder_padding_mask,
            head_mask=None,
        )
        hidden_states = stack_output[0]
        extra = None
        if self.training:
            encoder_moe_loss = encoder_out.moe_loss or []
            decoder_moe_loss = stack_output[-2]
            moe_loss = encoder_moe_loss + decoder_moe_loss # append list
            extra = {"moe_loss": moe_loss}
            if self.config.gate_logits:
                # encoder gate logit is not used because it has different
                # seq dimention and is not aligned
                # encoder_gate_logits = encoder_out.moe_gate_logits or []
                extra['gate_logits'] = stack_output[-1]

        return hidden_states, extra
