#!/usr/bin/env python3

# Copyright 2022 Electronics and Telecommunications Research Institute (Jeong-Uk, Bang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Create train, valid, eval subsets for switchboard-based backchannel detection
"""

import argparse
import json
import pdb
import sys


def main(args):
    """Run the main split function."""
    parser = argparse.ArgumentParser(description="Split subsets")
    parser.add_argument("-txt", help="all.txt2", required=True)
    parser.add_argument("-train", help="conversations.train", required=True)
    parser.add_argument("-valid", help="conversations.valid", required=True)
    parser.add_argument("-eval", help="conversations.eval", required=True)

    args = parser.parse_args()

    # read files
    fin_txt = open(args.txt, "r")
    fin_train = open(args.train, "r")
    fin_valid = open(args.valid, "r")
    fin_eval = open(args.eval, "r")

    fout_train = open('train.txt', 'w')
    fout_valid = open('valid.txt', 'w')
    fout_eval = open('eval.txt', 'w')

    # do partitioning
    tr, vs, es = [], [], []
    for line in fin_train:
        line = line.strip()
        tr.append(line)

    for line in fin_valid:
        line = line.strip()
        vs.append(line)

    for line in fin_eval:
        line = line.strip()
        es.append(line)

    # save subsets
    for line in fin_txt:
        line = line.strip()
        items = line.split()
        wav = items[0].split('/')[2].split('-')[0][:6]

        if wav in tr:
            fout_train.write(line+'\n')
        elif wav in vs:
            fout_valid.write(line+'\n')
        else:
            fout_eval.write(line+'\n')

   
if __name__ == "__main__":
    main(sys.argv[1:])
