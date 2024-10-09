import argparse
import os
import t5

# from google.cloud import storage

os.environ["NO_GCE_CHECK"] = "true"

def main(args):
    vocab = t5.data.get_default_vocabulary()
    sp = vocab.tokenizer


    with open(args.input, encoding='utf-8') as f, \
        open(args.output, 'w', encoding='utf-8') as out:
        for line in f:
            # out.write(' '.join(sp.encode(line)))
            out.write(' '.join(sp.encode(line, out_type=str)))
            out.write('\n')
        # out.write('\n'.join([(w + ' 1') for w in vocab_list]))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="text input path")
    parser.add_argument('--output', type=str, required=True, help="text output path")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
