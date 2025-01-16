#!/usr/bin/env python3

# Copyright 2022 Electronics and Telecommunications Research Institute (Jeong-Uk, Bang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""" 
Extract speech segments for backchannel detection
"""

import argparse
import json
import sys


def main(args):
    """Run the main function."""
    # txt: swb_ms98_trans_all.txt
    # json: utterance_is_backchannel.json
    # db: switchboard corpus dir
    parser = argparse.ArgumentParser(description="Extract wav segments")
    parser.add_argument("-txt", "--txt", help="Input file (swb_ms98_trans_all.txt)", required=True)
    parser.add_argument("-json", "--json", help="Input file (utterance_is_backchannel.json)", required=True)
    parser.add_argument("-db", "--db_dir", help="SWBD dirname", required=True)

    # file I/O
    args = parser.parse_args()
    fin_txt = open(args.txt, "r")
    with open(args.json, "r") as fin_json:
        jsons = json.load(fin_json)

    fout_sh = open('all_sphpipe.sh', 'w')
    fout_trans = open('all.trans', 'w')
    fout_bc = open('all.bc', 'w')
    fout_txt = open('all.txt', 'w')

    for line in fin_txt:
        # read data
        items = line.split()
        fn, st, et, trans = items[0], items[1], items[2], " ".join(items[3:])
        dialog = fn.split('-')[0]
        wav = dialog[:6]
        ch = '1' if dialog[6] == 'A' else '2'

        # add meta symbols "[sil]", "[lau"], "[spn]", "[nsn]"
        # remove symbols "<b_aside>" => "", "<e_aside>" => "", 
        trans = trans.replace('[silence]', '[sil]')
        trans = trans.replace('[laughter]', '[lau]').replace('[noise]', '[nsn]')
        trans = trans.replace('[vocalized-noise]', '[spn]')
        trans = trans.replace('<b_aside>', '').replace('<e_aside>', '')

        # backchannel symbols [mono-bc], [dialog-bc], [non-bc]
        bc = jsons[fn].replace('monologuing-bc', 'mono-bc')

        # write files
        fout_sh.write('sph2pipe -f wav -p -c ' + ch + ' -t ' + st + ':' + et +
                      ' ./swbd/'+wav+'.wav ./swbd_splited/'+fn+'.wav\n')
        fout_trans.write('./swbd_splited/'+fn+'.wav '+trans+'\n')
        fout_bc.write('./swbd_splited/'+fn+'.wav ['+bc+']\n')
        fout_txt.write('./swbd_splited/'+fn+'.wav '+ trans +' ['+bc+']\n')

   
if __name__ == "__main__":
    main(sys.argv[1:])
