import torch
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.t5.t5_model import T5Model
from .t5_moe_encoder import T5_MOE_Encoder
from .t5_moe_decoder import T5_MOE_Decoder


@register_model('t5_moe')
class T5_MOE_Model(T5Model):
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--gate-dim', type=int, metavar='N',
                            help='Dimension of the MoE gate')
        parser.add_argument('--gate-top-n', type=int, metavar='N',
                            help='Number of experts chosen for each token')
        parser.add_argument('--moe-type', type=str, 
                            help='Type of MoE module')
        parser.add_argument('--moe-freq', type=int, metavar='N',
                            help='Put MoE modules every n-th layers')
        parser.add_argument('--moe-location', type=str,
                            help='Where MoE modules should be applied. Can include combination of "encoder", "decoder", and ("logits1" or "logits2")')
        parser.add_argument('--num-experts', type=int, metavar='N',
                            help='Number of experts in each MoE module')
        parser.add_argument('--moe-scaling', type=int, default=1,
                            help='The multiplier of the MoE output')
        parser.add_argument('--expert-hidden', type=int, metavar='N',
                            help='The hidden size of the MoE experts')
        parser.add_argument('--expert-dropout', type=float, metavar='F',
                            help='Dropout for the expert layers')
        parser.add_argument('--share-expert-gelu', action='store_true', default=False,
                            help='Whether to share the GeLU parameters of the experts')
        parser.add_argument('--differentiable-topk', action='store_true',
                            help='Whether to use differentiable MoE gate')
        parser.add_argument('--share-gate', action='store_true', default=False,
                            help='Whether to use share the parameters of the MoE gate')
        parser.add_argument('--gate-logits', action='store_true', default=False,
                            help='Whether to get logits from the MoE gate')
        parser.add_argument('--gate-class-dim', type=int, default=None,
                            help='Whether to get logits from the MoE gate')
        parser.add_argument('--gate-hidden-dims', type=str, default=None,
                            help='The hidden dimensions of the MoE gate in the form of list with string data type (e.g., "[1,2]")')
        parser.add_argument('--gate-capacity', type=str, default="[1.2, 2.4]",
                            help='The capacity of the MoE gate for train and eval in the form of list with string data type (e.g., "[1.2,2.4]")')
        parser.add_argument('--merge-backbone', action='store_true', default=False,
                            help='Whether to merge the main backbone layer of the MoE layer')
        parser.add_argument('--freeze-backbone', action='store_true', default=False,
                            help='Whether to freeze the main backbone layer of the MoE layer')
        parser.add_argument('--freeze-non-MoE', action='store_true', default=False,
                            help='Whether to freeze the parameters of other than the MoE layers')
    
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return T5_MOE_Encoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return T5_MOE_Decoder(args, tgt_dict, embed_tokens, False)

    def load_state_dict(self, state_dict, strict=True, args=None):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # print({k: torch.norm(v) for k, v in state_dict.items() if 'DenseReluDense' in k})
        # uninitialized = {k: torch.norm(v) for k, v in model_dict.items() if k not in state_dict}
        # print("Uninitialized weights:", uninitialized)
        for k, v in model_dict.items():
            # assign experts that are not in the pre-trained model
            if args.moe_type not in ["hard", "soft"] and k not in state_dict:
                print("No weight loading with moe type {} for {}".format(args.moe_type, k))
            elif k not in state_dict and 'moe.experts' in k:
                if args.merge_backbone:
                    """
                    assign weights of
                    decoder.t5_stack.block.X.layer.2.moe.experts.Y.wi.0.weight
                    to
                    decoder.t5_stack.block.X.layer.2.DenseReluDense.wi.Z.weight +
                      decoder.t5_stack.block.X.layer.2.moe.experts.Y.htoh4.weight
                    """
                    name_dict_parts = k.split('.')
                    init_main_w = False
                    if name_dict_parts[-1] != 'bias':
                        name_dict_parts = k.split('.')
                        expert_idx = ''
                        main_weight_name = '.'.join(name_dict_parts[0:6]) + '.DenseReluDense.' + '.'.join(name_dict_parts[9:])
                        main_weight = state_dict[main_weight_name].unsqueeze(0)
                        assert v.shape == main_weight.shape, \
                            "shape of base layer ({}) is different from MoE expert layer ({})".format(
                                v.shape, main_weight.shape)
                        print('\nInitiating weights from {}:\n{}'.format(main_weight_name, main_weight))
                        print('\nUpdating weights of {} from:\n{} to:\n{}'.format(k, v, main_weight))
                        pretrained_dict[k] = main_weight.to(v.device)
                        init_main_w = True

                    exp_prefix = '.'.join(name_dict_parts[0:9])
                    layer_type = name_dict_parts[9]
                    if layer_type == 'wi':
                        if name_dict_parts[8] != '1':
                            # do not change the wi.0 of the pretrained model
                            continue
                        exp_type = 'htoh4.'
                    elif layer_type == 'wo':
                        exp_type = 'h4toh.'
                    else:
                        print('[WARNING] Unrecognized layer', k, flush=True)
                        continue
                    exp_weight_name = exp_prefix + '.' + exp_type + name_dict_parts[-1]
                    if exp_weight_name in state_dict:
                        exp_weight = state_dict[exp_weight_name]
                        assert v.shape == exp_weight.shape, \
                            "shape of base layer ({}) is different from MoE expert layer ({})".format(
                                v.shape, exp_weight.shape)
                        print('\nAdding weights from {}:\n{}'.format(exp_weight_name, exp_weight))
                        if init_main_w:
                            pretrained_dict[k] += exp_weight.to(v.device)
                            prev_w = pretrained_dict[k]
                            print('\nUpdating weights of {} from:\n{} to:\n{}'.format(k, prev_w, pretrained_dict[k]))
                        else:
                            pretrained_dict[k] = exp_weight.to(v.device)
                            print('\nInitiating weights of {} to:\n{}'.format(k, pretrained_dict[k]))
                    else:
                        if init_main_w:
                            pretrained_dict[k] += v.to(v.device)
                            print('Weight {} is not found, adding random weights \n{}.'.format(exp_weight_name, v))
                        else:
                            print('Weight {} is not found, keeping the random weights \n{}.'.format(exp_weight_name, v))
                else:
                    """
                    assign weights of
                    1. encoder.t5_stack.block.X.layer.1.moe.experts.experts.0.wi.Y.weight
                    2. encoder.t5_stack.block.X.layer.1.moe.experts.experts.1.wo.weight
                    to
                    1. encoder.t5_stack.block.X.layer.1.DenseReluDense.wi.Y.weight
                    2. encoder.t5_stack.block.0.layer.1.DenseReluDense.wo.weight
                    and
                    1. decoder.t5_stack.block.X.layer.2.moe.experts.experts.0.wi.Y.weight
                    2. decoder.t5_stack.block.X.layer.2.moe.experts.experts.1.wo.weight
                    to
                    1. decoder.t5_stack.block.X.layer.2.DenseReluDense.wi.Y.weight
                    2. decoder.t5_stack.block.X.layer.2.DenseReluDense.wo.weight
                    """
                    name_dict_parts = k.split('.')
                    expert_idx = ''
                    for idx in range(len(name_dict_parts)):
                        # the sequence is moe.experts.experts.{expert_idx}
                        if name_dict_parts[idx] == "experts" and \
                            name_dict_parts[idx - 1] == "experts" and \
                                name_dict_parts[idx - 2] == "moe":
                            expert_idx = name_dict_parts[idx + 1]
                            break
                    if len(expert_idx) == 0: #somehow?
                        expert_idx = name_dict_parts[9] # default
                    is_encoder = name_dict_parts[0] == 'encoder'
                    if 'moe_logit' in k:
                        if "logits2" in args.moe_location:
                            continue
                        elif "logits1" in args.moe_location:
                            mirroring_name = 'decoder.t5_stack.logits.weight'
                    else:
                        mirroring_name = '.'.join(name_dict_parts[0:6]) + '.DenseReluDense.'\
                            + '.'.join(name_dict_parts[10:])
                    ref_v = state_dict[mirroring_name]
                    if int(expert_idx) == 0:
                        new_weight = ref_v
                    else:
                        var = torch.var(ref_v)
                        noise = torch.rand_like(ref_v).uniform_(-var, var)
                        # noise = torch.rand_like(ref_v).uniform_(-1e-6, 1e-6)

                        # TODO: change back to adding noise
                        new_weight = ref_v.detach() + noise
                        # new_weight = ref_v.detach()
                    pretrained_dict[k] = new_weight
                    print('\nLoading weight from {}: {}'.format(mirroring_name, ref_v))
                    print("Expert {}\n assigned to \n{} from \n{}".format(k, new_weight, v))
            elif k not in state_dict and 'moe.ff_after' in k:
                pretrained_dict[k] = state_dict['decoder.t5_stack.logits.weight']
                print('\nLoading weight from {}: {}'.format('decoder.t5_stack.logits.weight', state_dict['decoder.t5_stack.logits.weight']))
                print("Expert {}\n assigned to \n{}".format(k, pretrained_dict[k]))
        model_dict.update(pretrained_dict) 
        return super().load_state_dict(model_dict, strict=strict)


@register_model_architecture('t5_moe', 't5-moe-small')
def t5_small_architecture(args):
    args.d_ff = getattr(args, 'd_ff', 2048)
    args.d_kv = getattr(args, 'd_kv', 64)
    args.d_model = getattr(args, 'd_model', 512)
    args.dropout_rate = getattr(args, 'dropout_rate', 0.1)
    args.layer_norm_epsilon = getattr(args, 'layer_norm_epsilon', 1e-06)
    args.n_positions = getattr(args, 'n_positions', 512)
    args.num_heads = getattr(args, 'num_heads', 8)
    args.num_layers = getattr(args, 'num_layers', 6)
    args.relative_attention_num_buckets = getattr(args, 'relative_attention_num_buckets', 32)
    args.vocab_size = getattr(args, 'vocab_size', 32128)
    args.tie_embeddings = getattr(args, 'tie_embeddings', True)
    args.special_chars = False
    args.gate_dim = getattr(args, 'gate_dim', args.d_model)
    args.gate_top_n = getattr(args, 'gate_top_n', 2)
    args.differentiable_topk = getattr(args, 'differentiable_topk', False)
    args.expert_dropout = getattr(args, 'expert_dropout', args.dropout_rate)
    args.moe_location = getattr(args, 'moe_location', "encoder-decoder")
    args.expert_hidden = getattr(args, 'expert_hidden', args.d_ff)
    args.gate_capacity = getattr(args, 'gate_capacity', args.gate_capacity)
    args.merge_backbone = getattr(args, 'merge_backbone', args.merge_backbone)

@register_model_architecture('t5_moe', 't5-moe-v1.1-small')
def t5_v1_1_small_architecture(args):
    args.activation_fn = getattr(args, 'activation_fn', 'gated-gelu')
    args.tie_embeddings = getattr(args, 'tie_embeddings', False)
    args.num_heads = getattr(args, 'num_heads', 6)
    args.num_layers = getattr(args, 'num_layers', 8)
    args.d_ff = getattr(args, 'd_ff', 1024)
    t5_small_architecture(args)


@register_model_architecture('t5_moe', 't5-moe-base')
def t5_base_architecture(args):
    args.d_ff = getattr(args, 'd_ff', 3072)
    args.d_kv = getattr(args, 'd_kv', 64)
    args.d_model = getattr(args, 'd_model', 768)
    args.dropout_rate = getattr(args, 'dropout_rate', 0.1)
    args.layer_norm_epsilon = getattr(args, 'layer_norm_epsilon', 1e-06)
    args.relative_attention_num_buckets = getattr(args, 'relative_attention_num_buckets', 32)
    args.n_positions = getattr(args, 'n_positions', 512)
    args.num_heads = getattr(args, 'num_heads', 12)
    args.num_layers = getattr(args, 'num_layers', 12)
    args.vocab_size = getattr(args, 'vocab_size', 32128)
    args.tie_embeddings = getattr(args, 'tie_embeddings', True)
    args.special_chars = False
    args.gate_dim = getattr(args, 'gate_dim', args.d_model)
    args.gate_top_n = getattr(args, 'gate_top_n', 2)
    args.differentiable_topk = getattr(args, 'differentiable_topk', False)
    args.expert_dropout = getattr(args, 'expert_dropout', args.dropout_rate)
    args.moe_location = getattr(args, 'moe_location', "encoder-decoder")
    args.expert_hidden = getattr(args, 'expert_hidden', args.d_ff)
    args.gate_capacity = getattr(args, 'gate_capacity', args.gate_capacity)
    args.merge_backbone = getattr(args, 'merge_backbone', args.merge_backbone)

@register_model_architecture('t5_moe', 't5-moe-v1.1-base')
def t5_v1_1_base_architecture(args):
    args.activation_fn = getattr(args, 'activation_fn', 'gated-gelu')
    args.tie_embeddings = getattr(args, 'tie_embeddings', False)
    args.d_ff = getattr(args, 'd_ff', 2048)
    args.d_kv = getattr(args, 'd_kv', 64)
    args.d_model = getattr(args, 'd_model', 768)
    t5_base_architecture(args)


@register_model_architecture('t5_moe', 't5-moe-large')
def t5_large_architecture(args):
    args.d_ff = getattr(args, 'd_ff', 4096)
    args.d_kv = getattr(args, 'd_kv', 64)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.d_model = getattr(args, 'd_model', 1024)
    args.dropout_rate = getattr(args, 'dropout_rate', 0.1)
    args.layer_norm_epsilon = getattr(args, 'layer_norm_epsilon', 1e-06)
    args.n_positions = getattr(args, 'n_positions', 512)
    args.num_heads = getattr(args, 'num_heads', 16)
    args.num_layers = getattr(args, 'num_layers', 24)
    args.relative_attention_num_buckets = getattr(args, 'relative_attention_num_buckets', 32)
    args.vocab_size = getattr(args, 'vocab_size', 32128)
    args.tie_embeddings = getattr(args, 'tie_embeddings', True)
    args.special_chars = False
    args.gate_dim = getattr(args, 'gate_dim', args.d_model)
    args.gate_top_n = getattr(args, 'gate_top_n', 2)
    args.differentiable_topk = getattr(args, 'differentiable_topk', False)
    args.expert_dropout = getattr(args, 'expert_dropout', args.dropout_rate)
    args.moe_location = getattr(args, 'moe_location', "encoder-decoder")
    args.expert_hidden = getattr(args, 'expert_hidden', args.d_ff)
    args.gate_capacity = getattr(args, 'gate_capacity', args.gate_capacity)
    args.merge_backbone = getattr(args, 'merge_backbone', args.merge_backbone)

@register_model_architecture('t5_moe', 't5-moe-v1.1-large')
def t5_v1_1_large_architecture(args):
    args.activation_fn = getattr(args, 'activation_fn', 'gated-gelu')
    args.tie_embeddings = getattr(args, 'tie_embeddings', False)
    args.d_ff = getattr(args, 'd_ff', 2816)
    args.d_kv = getattr(args, 'd_kv', 64)
    args.d_model = getattr(args, 'd_model', 1024)
    t5_large_architecture(args)


@register_model_architecture('t5_moe', 't5-moe-3b')
def t5_3b_architecture(args):
    args.d_ff = getattr(args, 'd_ff', 16384)
    args.d_kv = getattr(args, 'd_kv', 128)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.d_model = getattr(args, 'd_model', 1024)
    args.dropout_rate = getattr(args, 'dropout_rate', 0.1)
    args.layer_norm_epsilon = getattr(args, 'layer_norm_epsilon', 1e-06)
    args.n_positions = getattr(args, 'n_positions', 512)
    args.num_heads = getattr(args, 'num_heads', 32)
    args.num_layers = getattr(args, 'num_layers', 24)
    args.relative_attention_num_buckets = getattr(args, 'relative_attention_num_buckets', 32)
    args.vocab_size = getattr(args, 'vocab_size', 32128)
    args.tie_embeddings = getattr(args, 'tie_embeddings', True)
    args.special_chars = False
    args.gate_dim = getattr(args, 'gate_dim', args.d_model)
    args.gate_top_n = getattr(args, 'gate_top_n', 2)
    args.differentiable_topk = getattr(args, 'differentiable_topk', False)
    args.expert_dropout = getattr(args, 'expert_dropout', args.dropout_rate)
    args.moe_location = getattr(args, 'moe_location', "encoder-decoder")
    args.expert_hidden = getattr(args, 'expert_hidden', args.d_ff)
    args.gate_capacity = getattr(args, 'gate_capacity', args.gate_capacity)
    args.merge_backbone = getattr(args, 'merge_backbone', args.merge_backbone)

@register_model_architecture('t5_moe', 't5-moe-v1.1-xl')
def t5_v1_1_xl_architecture(args):
    args.activation_fn = getattr(args, 'activation_fn', 'gated-gelu')
    args.tie_embeddings = getattr(args, 'tie_embeddings', False)
    args.d_ff = getattr(args, 'd_ff', 5120)
    args.d_kv = getattr(args, 'd_kv', 64)
    args.d_model = getattr(args, 'd_model', 2048)
    t5_3b_architecture(args)


@register_model_architecture('t5_moe', 't5-moe-11b')
def t5_11b_architecture(args):
    args.d_ff = getattr(args, 'd_ff', 65536)
    args.d_kv = getattr(args, 'd_kv', 128)
    args.d_model = getattr(args, 'd_model', 1024)
    args.dropout_rate = getattr(args, 'dropout_rate', 0.1)
    args.layer_norm_epsilon = getattr(args, 'layer_norm_epsilon', 1e-06)
    args.n_positions = getattr(args, 'n_positions', 512)
    args.num_heads = getattr(args, 'num_heads', 128)
    args.num_layers = getattr(args, 'num_layers', 24)
    args.relative_attention_num_buckets = getattr(args, 'relative_attention_num_buckets', 32)
    args.vocab_size = getattr(args, 'vocab_size', 32128)
    args.tie_embeddings = getattr(args, 'tie_embeddings', True)
    args.special_chars = False
