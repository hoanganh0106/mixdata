import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True)
parser.add_argument('--output_csv', type=str, required=True)
parser.add_argument('--n', type=int, default=3000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--shuffle', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv, engine='python')
    if args.shuffle:
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    out_df = df.head(args.n)
    out_df.to_csv(args.output_csv, index=False)
    print(f'Wrote {len(out_df)} rows to {args.output_csv}')
