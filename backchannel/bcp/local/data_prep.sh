#!/bin/bash

# Copyright 2022 Electronics and Telecommunications Research Institute (Jeong-Uk, Bang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. utils/parse_options.sh || exit 1;

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <db-dir> <dst-dir>"
    echo "e.g.: $0 /bls/backchanneldb/swbd_bc data/train"
    exit 1
fi

db=$1
dst=$2

data=$(echo $dst | sed 's:\.:/:' | awk -v db=$db -F"/" '{print db"/scripts/"$NF".txt"}')
temp=tmp

mkdir -p ${dst} ${dst}/$temp || exit 1;

[ ! -d ${db} ] && echo "$0: no such directory ${db}" && exit 1;
[ ! -f ${data} ] && echo "$0: no such file ${data}. please check the SWBD corpus of ETRI version (jubang0219@etri.re.kr)." && exit 1;

wav_scp=${dst}/wav.scp; [[ -f "${wav_scp}" ]] && rm ${wav_scp}
text=${dst}/text; [[ -f "${text}" ]] && rm ${text}
utt2spk=${dst}/utt2spk; [[ -f "${utt2spk}" ]] && rm ${utt2spk}

# 1) extract meta data
cat $data | cut -f1 -d' ' > ${dst}/${temp}/wav_8k.list
cat ${dst}/${temp}/wav_8k.list | awk -F"/" '{print $NF}' | awk -F"." '{print $1}' > ${dst}/${temp}/labels
awk -v db=$db '{print db "/" $0}' ${dst}/${temp}/wav_8k.list | \
    paste -d' ' ${dst}/${temp}/labels - | sort > ${dst}/${temp}/wav_8k.scp

# 2) prepare wav.scp
awk '{print $1 " sox -R -t wav " $2 " -t wav - rate 16000 dither |"}' \
    ${dst}/${temp}/wav_8k.scp > ${dst}/wav.scp

# 3) prepare text
cat $data | cut -d' ' -f3- > ${dst}/${temp}/text.org
cat ${dst}/${temp}/text.org | paste -d' ' ${dst}/${temp}/labels - | sort > ${dst}/text

# 4) prepare utt2spk & spk2utt
spk2utt=${dst}/spk2utt
awk '{print $1 " " $1}' ${dst}/${temp}/labels | sort -k 1 > $utt2spk || exit 1
utils/utt2spk_to_spk2utt.pl < ${utt2spk} > $spk2utt || exit 1

ntext=$(wc -l <$text)
nutt2spk=$(wc -l <$utt2spk)
! [ "$ntext" -eq "$nutt2spk" ] && \
  echo "Inconsistent #transcripts($ntext) and #utt2spk($nutt2spk)" && exit 1

utils/validate_data_dir.sh --no-feats $dst || exit 1

echo "$0: successfully prepared data in ${dst}"

exit 0;
