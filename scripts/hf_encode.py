import argparse
# import sentencepiece as spm
from transformers import T5TokenizerFast


def main(args):
    if args.model_name is not None:
        tokenizer = T5TokenizerFast.from_pretrained(args.model_name, use_fast=True)
    else:
        tokenizer = T5TokenizerFast(args.tokenizer_model, use_fast=True)
    vocab = tokenizer.get_vocab()

    # old code: directly tokenize and write
    # with open(args.input, encoding='utf-8') as f, \
    #     open(args.output, 'w', encoding='utf-8') as out:
    #     for line in f:
    #         out.write(' '.join(tokenizer.tokenize(line)))
    #         out.write('\n')
    
    with open(args.input, encoding='utf-8') as f:
        inputs = [l.strip() for l in f.readlines()]
    
    if hasattr(args, "aux_inputs") and args.aux_inputs is not None:
        aux_inputs = []
        for aux_path in args.aux_inputs:
            with open(aux_path) as f:
                aux_input = [l.strip().split() for l in f.readlines()]
                assert len(aux_input) == len(inputs), \
                    "Aux Input and main input need to have the same number of sentences"
                for aux_line, input_line in zip(aux_input, inputs):
                    # need to have the same amount of words
                    assert len(aux_line) == len(input_line.split()), \
                        "Aux Input and main input need to have the same number of words"
                aux_inputs.append(aux_input) # one line is list of tokens
            assert len(aux_input) == len(inputs)
    else:
        aux_inputs = None
    

    tokenized = tokenizer(inputs, add_special_tokens=False, return_offsets_mapping=True)
    results = []
    for batch_id, token_ids in enumerate(tokenized['input_ids']):
        decoded = tokenizer.convert_ids_to_tokens(token_ids)
        if tokenizer.unk_token in decoded:
            fix_decoded = []
            for tok_id, token in enumerate(decoded):
                if token == tokenizer.unk_token:
                    charspan = tokenized.token_to_chars(batch_id, tok_id)
                    fix_decoded.append(inputs[batch_id][charspan.start:charspan.end])
                else:
                    fix_decoded.append(token)
            if args.verbose:
                print('fix unk character from \n{}\nto\n{}'.format(decoded, fix_decoded))
            decoded = fix_decoded
        results.append(decoded)
    
    with open(args.output, 'w', encoding='utf-8') as out:
        out.write('\n'.join([' '.join(r) for r in results]) + '\n')

    if aux_inputs is not None:
        aux_results = []
        for aux_input in aux_inputs:
            aux_result = []
            for batch_id in range(len(tokenized['input_ids'])):
                aux_line = []
                word_ids = tokenized.word_ids(batch_index=batch_id)
                for word_id in word_ids:
                    aux_line.append(aux_input[batch_id][word_id] if word_id is not None else 'KEEP')
                assert len(aux_line) == len(results[batch_id])
                aux_result.append(' '.join(aux_line))
            aux_results.append('\n'.join(aux_result) + '\n')
        
        assert len(aux_results) == len(args.aux_outputs)
        for aux_result, aux_output in zip(aux_results, args.aux_outputs):
            with open(aux_output, 'w', encoding='utf-8') as out:
                out.write(aux_result)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, help='huggingface model name')
    parser.add_argument('--tokenizer_model', type=str, default="spiece.model",
        help='path to the SentencePiece tokenizer model')
    parser.add_argument('--input', type=str, required=True, help="text input path")
    parser.add_argument('--output', type=str, required=True, help="text output path")
    parser.add_argument('--aux_inputs', type=str, nargs='*', help="auxiliary input path")
    parser.add_argument('--aux_outputs', type=str, nargs='*', help="auxiliary output path")
    parser.add_argument('--verbose', default=False, action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args)