import argparse
import os


def read_cat_report(texts):
    start_line = False
    cat = {}
    glob_stat_idx = None
    for l_idx, line in enumerate(texts):
        line = line.strip()
        if line == "===================== Span-Based Correction ======================":
            start_line = True
        elif line == "=========== Span-Based Correction ============":
            glob_stat_idx = l_idx + 2
            break
        elif len(line) == 0 or line.startswith("Category"):
            continue
        elif start_line:
            line_comps = line.split()
            assert len(line_comps) == 7, "Bad format: {}".format(line_comps)
            cat[line_comps[0]] = {
                'TP': float(line_comps[-6]),
                'FP': float(line_comps[-5]),
                'FN': float(line_comps[-4]),
                'P': float(line_comps[-3]),
                'R': float(line_comps[-2]),
                'F0.5': float(line_comps[-1]),
            }
    if glob_stat_idx is not None:
        line_comps = texts[glob_stat_idx].split()
        cat['X:global'] = {
                'TP': float(line_comps[-6]),
                'FP': float(line_comps[-5]),
                'FN': float(line_comps[-4]),
                'P': float(line_comps[-3]),
                'R': float(line_comps[-2]),
                'F0.5': float(line_comps[-1]),
            }
    return cat


def main(args):
    best_ckpt = None
    scores = []
    error_type = args.error_type or 'global'
    for c in os.listdir(args.report_dir):
        c_path = os.path.join(args.report_dir, c)
        if os.path.isfile(c_path):
            report_path = c_path
        elif os.path.isdir(c_path):
            errant_res = None
            errant_report = None
            for cc in os.listdir(c_path):
                if cc.endswith('errant-report'):
                    errant_report = cc
                elif cc.endswith('errant-res'):
                    errant_res = cc
            if errant_report is None and errant_res is None:
                print("[WARNING] report not found under ", cc)
                continue
            elif args.max_update is not None:
                try:
                    num_updates = c.split('.')[0].split('_')[2]
                except Exception as e:
                    print('[WARNING] skipping {} due to parsing issue'.format(
                        c))
                    continue
                if num_updates.isdigit() and int(num_updates) > args.max_update:
                    print("[WARNING] skipping {} (bigger than {})".format(
                        c, args.max_update))
                    continue
            report_path = os.path.join(c_path, errant_report or errant_res)
        else:
            print("[WARNING] failed to get info from", c)
            continue
    
        with open(report_path, encoding='utf-8') as f:
            r = read_cat_report(f.readlines())
            if len(r) == 0:
                print("[WARNING] failed to get info from", report_path)
                continue
        if args.scope == 'macro':
            cur_scores = []
            for k, v in r.items():
                k = k[2:] # ignore operation type
                if k.startswith(error_type):
                    cur_scores.append(v['F0.5'])
            
            if args.metric in ["avg", "average", "mean"]:
                ckpt_score = sum(cur_scores) / len(cur_scores)
            else:
                raise NotImplementedError(
                    "No implementation for metric {}".format(args.metric))
        elif args.scope == 'micro':
            cur_scores = {'TP': 0, 'FP': 0, 'FN': 0}
            for k, v in r.items():
                k = k[2:] # ignore operation type
                if k.startswith(error_type):
                    for c_k, _ in cur_scores.items():
                        cur_scores[c_k] += v[c_k]
            p = cur_scores['TP'] / (cur_scores['TP'] + cur_scores['FP'])
            r = cur_scores['TP'] / (cur_scores['TP'] + cur_scores['FN'])
            ckpt_score = ((1 + 0.5*0.5) * p * r) / ((0.5*0.5*p) + r)

        scores.append((ckpt_score, c))
    
    scores = sorted(scores, reverse=True)
    print(scores[0])


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report_dir', type=str, default='.', help="")
    parser.add_argument('--error_type', type=str, default=None, help="")
    parser.add_argument('--max_update', type=int, default=None, help="")
    parser.add_argument('--metric', type=str, default="avg", help="")
    parser.add_argument('--scope', type=str, default="micro", help="")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
