import argparse
from collections import OrderedDict
import logging
import os
import re
import sys

import numpy as np
import tensorflow as tf
import torch

from fairseq.models.t5.t5_model import *
from fairseq.optim.adafactor import Adafactor

logger = logging.getLogger('__main__')
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tf-checkpoint', '-c',
        type=str,
        required=True)
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True)
    parser.add_argument(
        '--arch', '-a',
        type=str,
        choices=['t5-small', 't5-v1.1-small', 't5-base', 't5-v1.1-base', 't5-large', 't5-v1.1-large', 't5-3B', 't5-v1.1-xl', 't5-11B'],
        required=True)
    return parser.parse_args()

def load_tf_weights_in_t5(tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))

    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = OrderedDict()
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array
    
    model = OrderedDict()
    optimizer = OrderedDict()

    for txt_name in tf_weights.keys():
        name = txt_name.split("/")
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            print(name, tf_weights[txt_name])
            logger.info("Skipping {}".format("/".join(name)))
            continue
        if "_slot_" in name[-1]:
            last = name[-1]
            name.pop()
            name.append(last[:last.find('_slot_')])
            name.append(last[last.find('_slot_'):])
            out_name = []
            for m_name in name:
                if m_name in ('encoder', 'decoder'):
                    out_name.extend([m_name, 't5_stack'])
                elif re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                    scope_names = m_name.split('_')
                    out_name.extend([scope_names[0], str(int(scope_names[1]))])
                elif m_name in ["kernel", "scale", "embedding"]:
                    pass
                else:
                    out_name.append(m_name)

            last = out_name[-1]
            if 'embedding' not in name:
                if last == '_slot_vc':
                    out_name[-1] = 'exp_avg_sq_col'
                elif last == '_slot_vr':
                    out_name[-1] = 'exp_avg_sq_row'
                elif last == '_slot_v':
                    out_name[-1] = 'exp_avg_sq'
            else:
                if last == '_slot_vc':
                    out_name[-1] = 'exp_avg_sq_row'
                elif last == '_slot_vr':
                    out_name[-1] = 'exp_avg_sq_col'
                elif last == '_slot_v':
                    out_name[-1] = 'exp_avg_sq'

            optimizer['.'.join(out_name)] = torch.from_numpy(tf_weights[txt_name].astype(np.float32))
            continue

        out_name = []
        for m_name in name:
            if m_name in ('encoder', 'decoder'):
                out_name.extend([m_name, 't5_stack'])
            elif re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = m_name.split('_')
                out_name.extend([scope_names[0], str(int(scope_names[1]))])
            elif m_name in ["kernel", "scale", "embedding"]:
                out_name.append("weight")
            else:
                out_name.append(m_name)
        if out_name[-1] != 'weight':
            out_name.append("weight")

        if 'embedding' not in name:
            array = np.transpose(tf_weights[txt_name])
        else:
            array = tf_weights[txt_name]
        array = torch.from_numpy(array.astype(np.float32))
        
        model['.'.join(out_name)] = array
    return model, optimizer

def prepare_adafactor_last_state(model, optimizer_weights, global_steps):
    matrix_params = list(filter(lambda p: p.requires_grad and p.dim() == 2, model.parameters()))
    vector_params = list(filter(lambda p: p.requires_grad and p.dim() == 1, model.parameters()))
    param_groups = [
        {'params': matrix_params},
        {'params': vector_params},
    ]
    optimizer = Adafactor(param_groups)
    optimizer_state = optimizer.state_dict()

    optimizer_state['state'] = {}
    for params, group_name in zip(param_groups, ('matrices', 'vectors')):
        last_state = OrderedDict()
        for param in params['params']:
            param_id = id(param)
            param_state = OrderedDict()
            param_state['step'] = global_steps
            for x in model.state_dict():
                try:
                    if torch.all(torch.eq(model.state_dict()[x], param)):
                        param_name = x[:-7]
                except:
                    continue
            ada_vectors = list(filter(lambda x: param_name in x, optimizer_weights))
            for v_name in ada_vectors:
                param_state[v_name.split('.')[-1]] = optimizer_weights[v_name]
            param_state['RMS'] = 1.0
            last_state[param_id] = param_state
        optimizer_state['state'][group_name] = last_state

    return optimizer_state

def mock_task():
    task = argparse.Namespace()
    task.source_dictionary = None
    task.target_dictionary = None
    return task

def create_fairseq_checkpoint(model_weights, args, optimizer_state, global_steps):
    model = {}
    model['model']  = model_weights
    model['args'] = args
    model['last_optimizer_state'] = optimizer_state
    model['optimizer_history'] = \
        [{'criterion_name': 'CrossEntropyCriterion',
          'optimizer_name': 'FairseqAdafactor',
          'lr_scheduler_state': {'best': 4.667},
          'num_updates': global_steps}]
    model['extra_state'] = {
        'train_iterator': {
            'epoch': 1,
            'iterations_in_epoch': global_steps,
            'shuffle': True},
        'val_loss': 4.667,
        'best': 4.667,
    }
    return model


def get_model_args(arch):
    args = argparse.Namespace()
    if arch == 't5-small':
        t5_small_architecture(args)
    elif arch == 't5-v1.1-small':
        t5_v1_1_small_architecture(args)
    elif  arch == 't5-base':
        t5_base_architecture(args)
    elif  arch == 't5-v1.1-base':
        t5_v1_1_base_architecture(args)
    elif  arch == 't5-large':
        t5_large_architecture(args)
    elif  arch == 't5-v1.1-large':
        t5_v1_1_large_architecture(args)
    elif  arch == 't5-3B':
        t5_3b_architecture(args)
    elif  arch == 't5-v1.1-xl':
        t5_v1_1_xl_architecture(args)
    elif  arch == 't5-11B':
        t5_11b_architecture(args)

    args.load_weights = False
    args.arch = arch
    args.optimizer = 'adafactor'
    args.criterion = 'cross_entropy'
    return args

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, pytorch_dump_path, arch):
    model_weights, optimizer_weights = load_tf_weights_in_t5(tf_checkpoint_path)
    
    model_weights['encoder.t5_stack.embed_tokens.weight'] = model_weights['shared.weight']
    model_weights['decoder.t5_stack.embed_tokens.weight'] = model_weights['shared.weight']
    model_weights['decoder.embed_tokens.weight'] = model_weights['shared.weight']

    args = get_model_args(arch)
    task = mock_task()
    global_steps = int(tf_checkpoint_path.split('-')[-1])

    t5 = T5Model.build_model(args, task)
    t5.load_state_dict(model_weights)

    optimizer_state = prepare_adafactor_last_state(t5, optimizer_weights, global_steps)

    checkpoint = create_fairseq_checkpoint(model_weights, args, optimizer_state, global_steps)

    # Save pytorch-model
    print('Save PyTorch model to {}'.format(pytorch_dump_path))
    torch.save(checkpoint, pytorch_dump_path)


if __name__ == '__main__':
    args = parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint, args.output, args.arch)
