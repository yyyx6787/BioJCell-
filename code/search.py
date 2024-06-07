import os
import argparse
parser = argparse.ArgumentParser(description='Cell Division Jelly Training')
parser.add_argument('--data', type=str, default='wet', help='data')
parser.add_argument('--model', type=str, default='fno', help='data')
args = parser.parse_args()
ks = [1,3,5,7,9]

for k in ks:
    strs = 'python /wanghaixin/Cell_Division/{}_{}.py --k {}'.format(args.model, args.data,int(k))
    os.system(strs)


