#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''


@Time    :   2022/06/13 21:55:14
@Author  :   Ma (Ma787639046@outlook.com)
'''

import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        default=None,
                        required=True)
    args = parser.parse_args()
    # Remove tevatron temp
    remove_names = ['dev_ranks', 'encoding', 'train_ranks', 'train-hn', 'dev.rank.tsv.marco',
    's1_dev.rank.tsv.marco', 'train.rank.tsv']
    if os.path.exists(args.dir):
        for _file in remove_names:
            filepath = os.path.join(args.dir, _file)
            if os.path.exists(filepath):
                os.system(f"rm -rf {filepath}")
    return

if __name__ == '__main__':
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    main()

