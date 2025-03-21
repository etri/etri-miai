#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh || exit 1
. ./cmd.sh || exit 1
. ./db.sh || exit 1

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

dataset=
train_set=
valid_set=
test_set=

#data
bkbd_datadir=${BKBD}
ckbd_datadir=${CKBD}
swbd_datadir=${SWBD}
outputdir=data_local
set_version=
# espnet_datasets
#  |_ audios/{dataset}/{utt2id}.wav
#  |_ splits/{dataset}/{split}.text
#  |_ splits/{dataset}/{split}.label

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

if [ "${dataset}" == "bkbd" ]; then
    log "AIHUB Backchannel Data Preparation"
    # copy transcription and label files
    for x in train valid test; do
        mkdir -p ${outputdir}/bkbd/${x} && cp -a ${bkbd_datadir}/${set_version}/splits/${x}.text ${outputdir}/bkbd/${x}/text && cp -a ${bkbd_datadir}/${set_version}/splits/${x}.label ${outputdir}/bkbd/${x}/bc_label
    done

    for x in ${train_set} ${valid_set} ${test_set}; do
        local/data_prep.sh ${bkbd_datadir}/${set_version} ${outputdir}/bkbd data/${x}
    done

    # old version
    # local/trans_prep.sh ${datadir} data/local/ckbd

    
    # copy transcription and label files
    # for x in train valid test; do
    #     mkdir -p ${outputdir}/bkbd/${x} && cp -a ${bkbd_datadir}/splits/AIHub.SMT2023/${x}.text ${outputdir}/bkbd/${x}/text && cp -a ${bkbd_datadir}/splits/AIHub.SMT2023/${x}.label ${outputdir}/bkbd/${x}/bc_label
    # done

    # for x in ${train_set} ${valid_set} ${test_set}; do
    #     local/data_prep.sh ${bkbd_datadir} ${outputdir}/bkbd data/${x}
    # done
    # old version end
elif [ "${dataset}" == "ckbd" ]; then
    log "Counseling Backchannel Data Preparation"

    # copy transcription and label files
    # for x in train valid test; do
    #     mkdir -p ${outputdir}/ckbd/${x}
    #     for y in Counseling.SelectStar2022 Counseling.SelectStar2023 Counseling.SMT2023; do
    #         if [ ! -d ${ckbd_datadir}/splits/${y} ]; then
    #             log "Error: ${ckbd_datadir}/splits/${y} does not exist. skipping..."
    #             continue
    #         fi
    #     done
    #     cat ${ckbd_datadir}/splits/Counseling.*/${x}.text >${outputdir}/ckbd/${x}/text
    #     cat ${ckbd_datadir}/splits/Counseling.*/${x}.label >${outputdir}/ckbd/${x}/bc_label
    # done
    # for x in ${train_set} ${valid_set} ${test_set}; do
    #     local/data_prep.sh ${ckbd_datadir} ${outputdir}/ckbd data/${x}
    # done

    # copy transcription and label files
    for x in train valid test; do
        mkdir -p ${outputdir}/ckbd/${x} && cp -a ${ckbd_datadir}/${set_version}/splits/${x}.text ${outputdir}/ckbd/${x}/text && cp -a ${ckbd_datadir}/${set_version}/splits/${x}.label ${outputdir}/ckbd/${x}/bc_label
    done

    for x in ${train_set} ${valid_set} ${test_set}; do
        local/data_prep.sh ${ckbd_datadir}/${set_version} ${outputdir}/ckbd data/${x}
    done
elif [ "${dataset}" == "swbd" ]; then
    log "Switchboard Data Preparation"

    # copy transcription and label files
    for x in train valid test; do
        mkdir -p ${outputdir}/swbd/${x} && cp -a ${swbd_datadir}/${set_version}/splits/${x}.text ${outputdir}/swbd/${x}/text && cp -a ${swbd_datadir}/${set_version}/splits/${x}.label ${outputdir}/swbd/${x}/bc_label
    done

    for x in ${train_set} ${valid_set} ${test_set}; do
        local/data_prep.sh ${swbd_datadir}/${set_version} ${outputdir}/swbd data/${x}
    done

else
    log "Error: --dataset ${dataset} is not supported."
    exit 2
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
