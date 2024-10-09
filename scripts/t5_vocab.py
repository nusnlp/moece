import argparse
import t5

# from google.cloud import storage

def main(args):
    vocab = t5.data.get_default_vocabulary()
    sp = vocab.tokenizer

    size = sp.get_piece_size()
    vocab_list = []
    for _id in range(size):
        vocab_list.append(sp.id_to_piece(_id))
    # print(sp.id_to_piece(_id), 1)

    with open(args.output, 'w', encoding='utf-8') as out:
        out.write('\n'.join([(w + ' 1') for w in vocab_list]))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True, help="vocab output path")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
