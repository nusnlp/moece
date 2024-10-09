import argparse

import errant


def main(args):
    annotator = errant.load('en')
    with open(args.source_text, encoding='utf-8') as f:
        sources = f.readlines()
    with open(args.target_text, encoding='utf-8') as f:
        targets = f.readlines()
    
    edit_lines = []
    for source_str, target_str in zip(sources, targets):
        source_str = source_str.strip()
        target_str = target_str.strip()
        source = annotator.parse(source_str)
        target = annotator.parse(target_str)
        edits = annotator.annotate(source, target)
        
        if args.align == 'target':
            edit_len = len(target_str.split())
            start_atr = 'c_start'
            end_atr = 'c_end'
        elif args.align == 'source':
            edit_len = len(source_str.split())
            start_atr = 'o_start'
            end_atr = 'o_end'
        else:
            raise ValueError("--align {} is not recognized.".format(args.align))

        edit_types = ['KEEP'] * edit_len
        for e in edits:
            start = getattr(e, start_atr)
            end = getattr(e, end_atr)
            for e_idx in range(start, end):
                edit_type = e.type
                if not args.operation_tag:
                    edit_type = edit_type.split(':', 1)[1]
                edit_types[e_idx] = edit_type
        edit_lines.append(' '.join(edit_types))
    
    with open(args.output_path, 'w', encoding='utf-8') as out:
        out.write('\n'.join(edit_lines))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_text', type=str, required=True)
    parser.add_argument('--target_text', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--align', type=str, default='target', choices=['source', 'target'])
    parser.add_argument('--operation_tag', default=False, action='store_true', help='include U: R: M: suffix in error type')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
