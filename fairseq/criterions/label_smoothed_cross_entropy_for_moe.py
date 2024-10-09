# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy_for_moe')
class LabelSmoothedCrossEntropyCriterionForMoE(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, aux_smoothing, downsample_aux, downsample_label, aux_weight, moe_weight):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.aux_eps = aux_smoothing
        self.aux_w = aux_weight
        self.alpha = moe_weight
        self.downsample_aux = downsample_aux
        self.downsample_label = downsample_label

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--downsample-aux', default=None, type=float, metavar='D',
                            help='downsample rate for most frequent aux label')
        parser.add_argument('--downsample-label', default=None, type=int, metavar='D',
                            help='denotes which label needs to be downsampled')
        parser.add_argument('--aux-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--aux-weight', default=0., type=float, metavar='D',
                            help='alpha for MoE loss')
        parser.add_argument('--moe-weight', default=0., type=float, metavar='D',
                            help='alpha for MoE loss')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, moe_loss, all_aux_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'moe_loss': moe_loss.data,
            'all_aux_loss': all_aux_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def downsample_aux_label(self, label, ignore_index, downsample_rate=0.8, downsample_label=None):
        if downsample_label is None:
            downsample_label = torch.mode(label, keepdim=True).values
        mask = (label != downsample_label)
        downsampler = torch.rand_like(label, dtype=torch.float32) > downsample_rate
        mask = torch.logical_or(mask, downsampler).long()
        replacement = mask * ignore_index
        down_sampled_label = label * mask + replacement
        # down_sampled_label = label.masked_fill_(mask * downsampler, ignore_index)
        # print('downsampler', downsampler, mask * downsampler)
        # print('After downsampling', label, '\n', down_sampled_label, flush=True)
        return down_sampled_label


    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        if net_output[1] is not None and 'moe_loss' in net_output[1]:
            moe_loss = sum(net_output[1]['moe_loss']) / len(net_output[1]['moe_loss'])
            # most likely not needed for other than FixedGate
            # if reduce:
            #     moe_loss = moe_loss.sum()
            loss = loss + self.alpha * moe_loss
        else:
            moe_loss = torch.zeros_like(loss)
            
        if 'aux_target' in sample:
            assert net_output[1] is not None and 'gate_logits' in net_output[1]
            all_aux_loss = None
            num_gates = float(len(net_output[1]['gate_logits']))
            for gate_logit in net_output[1]['gate_logits']:
                aux_target = sample['aux_target']
                if self.downsample_aux is not None:
                    # print('aux_target before', aux_target, flush=True)
                    aux_target = self.downsample_aux_label(aux_target, self.padding_idx, self.downsample_aux, self.downsample_label)
                    # print('aux_target after', aux_target, flush=True)
                aux_target = aux_target.view(-1, 1)
                assert aux_target.shape[:-1] == gate_logit.shape[:-1], \
                    "Shapes of aux target ({}) and gate logits ({}) are different!".format(aux_target.shape[:-1], gate_logit.shape[:-1])
                gate_logit = F.log_softmax(gate_logit.float(), dim=-1)
                aux_loss, _aux_nll_loss = label_smoothed_nll_loss(
                    gate_logit, aux_target, self.aux_eps, ignore_index=self.padding_idx, reduce=reduce,
                )
                if all_aux_loss is None:
                    all_aux_loss = aux_loss
                else:
                    all_aux_loss = all_aux_loss + aux_loss
            loss = loss + (self.aux_w * all_aux_loss / num_gates)
        else:
            all_aux_loss = torch.zeros_like(loss)
        
        return loss, nll_loss, moe_loss, all_aux_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        moe_loss_mean = sum(log.get('moe_loss', 0) for log in logging_outputs) / len(logging_outputs)
        all_aux_sum = sum(log.get('all_aux_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('moe_loss', moe_loss_mean, 1, round=3)
        metrics.log_scalar('aux_loss', all_aux_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
