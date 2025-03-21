#!/usr/bin/env bash

# Copyright 2020 Electronics and Telecommunications Research Institute (Jeong-Uk, Bang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. utils/parse_options.sh || exit 1

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <db-dir> <src-dir> <dst-dir>"
    echo "e.g.: $0 data_pool/ETRI_Backchannel_Corpus/processed/espnet_datasets data/local/bkbd data/bkbd/train"
    exit 1
fi

db=$1
src=$2
dst=$3

data=$(echo $dst | awk -v src=$src -F"_" '{print src"/"$NF"/text"}')
label_data=$(echo $dst | awk -v src=$src -F"_" '{print src"/"$NF"/bc_label"}')

[ -d ${dst} ] && rm -rf ${dst}

mkdir -p ${dst} || exit 1

[ ! -d ${db} ] && echo "$0: no such directory ${db}" && exit 1
[ ! -f ${data} ] && echo "$0: no such file ${data}. please re-run the script of 'local/trans_prep.sh'." && exit 1

wav_scp=${dst}/wav.scp
[[ -f "${wav_scp}" ]] && rm ${wav_scp}
text=${dst}/text
[[ -f "${text}" ]] && rm ${text}
label=${dst}/bc_label
[[ -f "${label}" ]] && rm ${label}
utt2spk=${dst}/utt2spk
[[ -f "${utt2spk}" ]] && rm ${utt2spk}

# 1) extract audio path and copy text and bc_label
cat $data | cut -f1 -d' ' | awk -v db=$db '{print $0 " " db"/audios/"$0".wav"}' | sort >${dst}/wav.scp  # data manager version
# cat $data | cut -f1 -d' ' | awk -v db=$db -F"-" '{print $0 " " db"/audios/"$1"/"$0".wav"}' | sort >${dst}/wav.scp # old korean version
# cat $data | cut -f1 -d' ' | awk -v db=$db -F"-" '{print $0 " " db"/audios/SWBD/"$0".wav"}' | sort >${dst}/wav.scp # old swbd version
cat $data | sort >${dst}/text
cat $label_data | sort >${dst}/bc_label

# 2) prepare utt2spk & spk2utt
spk2utt=${dst}/spk2utt
cat $data | cut -f1 -d' ' | awk '{print $1 " " $1}' | sort -k 1 >$utt2spk || exit 1
utils/utt2spk_to_spk2utt.pl <${utt2spk} >$spk2utt || exit 1

ntext=$(wc -l <$text)
nutt2spk=$(wc -l <$utt2spk)
! [ "$ntext" -eq "$nutt2spk" ] && echo "Inconsistent #transcripts($ntext) and #utt2spk($nutt2spk)" && exit 1

utils/validate_data_dir.sh --no-feats $dst || exit 1

echo "$0: successfully prepared data in ${dst}"
exit 0
