#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
    local a b
    a=$1
    for b in "$@"; do
        if [ "${b}" -le "${a}" ]; then
            a="${b}"
        fi
    done
    echo "${a}"
}

SECONDS=0

# General configuration
stage=1                                               # Processes starts from the specified stage.
stop_stage=10000                                      # Processes is stopped at the specified stage.
skip_stages=                                          # Spicify the stage to be skipped
skip_data_prep=false                                  # Skip data preparation stages.
skip_train=false                                      # Skip training stages.
skip_eval=false                                       # Skip decoding and evaluation stages.
skip_upload=true                                      # Skip packing and uploading to zenodo
skip_upload_hf=true                                   # Skip uploading to hugging face stages.
eval_valid_set=false                                  # Run decoding for the validation set
ngpu=1                                                # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1                                           # The number of nodes.
nj=32                                                 # The number of parallel jobs.
inference_nj=32                                       # The number of parallel jobs in decoding.
gpu_inference=false                                   # Whether to perform gpu decoding.
dumpdir=bc_dump # Directory to dump features.
expdir=bc_exp   # Directory to save experiments.
python=python3                                        # Specify python to execute espnet commands.

# Data preparation related
local_data_opts=              # The options given to local/data.sh.
post_process_local_data_opts= # The options given to local/data.sh for additional processing in stage 3.

# Feature extraction related
feats_type=raw                     # Feature type (raw, raw_copy, fbank_pitch, or extracted).
audio_format=flac                  # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
multi_columns_input_wav_scp=false  # Enable multi columns mode for input wav.scp for format_wav_scp.py
multi_columns_output_wav_scp=false # Enable multi columns mode for output wav.scp for format_wav_scp.py
fs=16k                             # Sampling rate.
min_wav_duration=0.1               # Minimum duration in second.
max_wav_duration=20                # Maximum duration in second.

# Tokenization related
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole

# Backchannel model related
bc_task="bcp" # BC task mode. Either 'bcp' or ''.
bc_tag=       # Suffix to the result dir for backchannel model training.
bc_exp=       # Specify the directory path for BCP experiment.
# If this option is specified, bc_tag is ignored.
bc_stats_dir= # Specify the directory path for BCP statistics.
bc_config=    # Config for backchnnel model training.
bc_args=      # Arguments for asr model training, e.g., "--max_epoch 10".
# Note that it will overwrite args in asr config.
pretrained_model=          # Pretrained model to load
ignore_init_mismatch=false # Ignore initial mismatch
feats_normalize=global_mvn # Normalizaton layer type.
num_ref=1                  # Number of references for training.
# In supervised learning based speech enhancement / separation, it is equivalent to number of speakers.
num_inf= # Number of inferences output by the model

# Decoding related
use_streaming=false # Whether to use streaming decoding

batch_size=1
inference_tag=                                   # Suffix to the result dir for decoding.
inference_config=                                # Config for decoding.
inference_args=                                  # Arguments for decoding, e.g., "--lm_weight 0.1".
inference_bc_model=valid.macro_f1_scores.ave.pth # BCP model path for decoding.
# e.g.
# inference_bc_model=train.loss.best.pth
# inference_bc_model=3epoch.pth
# inference_bc_model=valid.acc.best.pth
# inference_bc_model=valid.loss.ave.pth
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=                # Name of training set.
valid_set=                # Name of validation set used for monitoring/tuning network training.
test_sets=                # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
label_train_text=         # Text file path of backchannel label training set.
nlsyms_txt=none           # Non-linguistic symbol list if existing.
hyp_cleaner=none          # Text cleaner for hypotheses (may be used with external tokenizers)
cleaner=none              # Text cleaner.
g2p=none                  # g2p method (needed if token_type=phn).
lang=noinfo               # The language type of corpus.
score_opts=               # The options given to sclite scoring
local_score_opts=         # The options given to local/score.sh.
bc_speech_fold_length=800 # fold_length for speech data during BCP training.
bc_text_fold_length=150   # fold_length for text data during BCP training.
bc_label_fold_length=150  # fold_length for label data during BCP training.

block_mapping=false
dataset_name=ckbd
dataset_type=v3
only_encoder=false
fronted_opts=
category_label=merge
token_type=char
token_list=
bpemodel=none
use_wandb=false
wandb_project=
wandb_name=

help_message=$(
    cat <<EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_stages    # Spicify the stage to be skipped (default="${skip_stages}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --skip_upload_hf    # Skip packing and uploading stages (default="${skip_upload_hf}").
    --eval_valid_set # Run decoding for the validation set (default="${eval_valid_set}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --feats_type       # Feature type (raw, raw_copy, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw or raw_copy, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").

    # BCP model related
    --bc_task         # ASR task mode. Either 'bcp' or ''. (default="${bc_task}").
    --bc_tag          # Suffix to the result dir for bcp model training (default="${bc_tag}").
    --bc_exp          # Specify the directory path for BCP experiment.
                       # If this option is specified, bc_tag is ignored (default="${bc_exp}").
    --bc_stats_dir    # Specify the directory path for BCP statistics (default="${bc_stats_dir}").
    --bc_config       # Config for bcp model training (default="${bc_config}").
    --bc_args         # Arguments for bcp model training (default="${bc_args}").
                       # e.g., --bc_args "--max_epoch 10"
                       # Note that it will overwrite args in bcp config.
    --pretrained_model=          # Pretrained model to load (default="${pretrained_model}").
    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --feats_normalize  # Normalizaton layer type (default="${feats_normalize}").
    --num_ref    # Number of references for training (default="${num_ref}").
                 # In supervised learning based speech recognition, it is equivalent to number of speakers.
    --num_inf    # Number of inference audio generated by the model (default="${num_inf}")
                 # Note that if it is not specified, it will be the same as num_ref. Otherwise, it will be overwritten.

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_bc_model # BCP model path for decoding (default="${inference_bc_model}").
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").
    --use_streaming       # Whether to use streaming decoding (default="${use_streaming}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --lang          # The language type of corpus (default=${lang}).
    --bc_speech_fold_length # fold_length for speech data during ASR training (default="${bc_speech_fold_length}").
    --bc_label_fold_length   # fold_length for text data during ASR training (default="${bc_label_fold_length}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
# shellcheck disable=SC2086
run_args=$(scripts/utils/print_args.sh $0 "$@")
# shellcheck disable=SC1091
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

# shellcheck disable=SC1091
. ./path.sh
# shellcheck disable=SC1091
. ./cmd.sh

# Define Variable
dumpdir=./dump_and_exp/${dataset_type}.dump
expdir=./dump_and_exp/${dataset_type}.exp

train_set=${dataset_name}/${dataset_type}.proc_train
valid_set=${dataset_name}/${dataset_type}.proc_valid
test_sets=${dataset_name}/${dataset_type}.proc_test

local_data_opts="--dataset ${dataset_name} --outputdir data_lc/${dataset_name}/${dataset_type} --train_set ${train_set} --valid_set ${valid_set} --test_set ${test_sets} --set_version ${dataset_type}"

# Check required arguments
if ! "${skip_train}"; then
    [ -z "${train_set}" ] && {
        log "${help_message}"
        log "Error: --train_set is required"
        exit 2
    }
    [ -z "${valid_set}" ] && {
        log "${help_message}"
        log "Error: --valid_set is required"
        exit 2
    }
fi
if ! "${eval_valid_set}"; then
    [ -z "${test_sets}" ] && {
        log "${help_message}"
        log "Error: --test_sets is required"
        exit 2
    }
else
    [ -z "${valid_set}" ] && {
        log "${help_message}"
        log "Error: --valid_set is required"
        exit 2
    }
fi

if [ -n "${train_set}" ] && [ "${train_set}" = "${valid_set}" ]; then
    log "Error: train_set and valid_set must be different. --train_set ${train_set} --valid_set ${valid_set}"
    exit 1
fi

_test_sets=
for dset in ${test_sets}; do
    if [ "${dset}" = "${train_set}" ]; then
        log "Error: train_set and test_sets must be different. --train_set ${train_set} --test_sets ${test_sets}"
        exit 1
    fi
    if [ "${dset}" = "${valid_set}" ]; then
        log "Info: The valid_set '${valid_set}' is included in the test_sets. '--eval_valid_set true' is set and '${valid_set}' is removed from the test_sets"
        eval_valid_set=true
    elif [[ " ${_test_sets} " =~ [[:space:]]${dset}[[:space:]] ]]; then
        log "Info: ${dset} is duplicated in the test_sets. One is removed"
    else
        _test_sets+="${dset} "
    fi
done
test_sets=${_test_sets}

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = raw_copy ]; then
    # raw_copy is as same as raw except for skipping the format_wav stage
    data_feats=${dumpdir}/raw_copy
elif [ "${feats_type}" = fbank_pitch ]; then
    data_feats=${dumpdir}/fbank_pitch
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" == extracted ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

num_inf=${num_inf:=${num_ref}}
# Preprocessor related
if [ "${num_ref}" -eq 1 ]; then
    # For single speaker, text file path and name are text
    ref_text_files_str="text "
    ref_text_names_str="text "
    ref_bc_label_files_str="bc_label "
    ref_bc_label_names_str="classification "
else
    # For multiple speakers, text file path and name are text_spk[1-N] and [text, text_spk2, ...]
    #TODO(simpleoier): later to support flexibly defined text prefix
    ref_text_files_str="text_spk1 "
    ref_text_names_str="text "
    ref_bc_label_files_str="bc_label "
    ref_bc_label_names_str="classification "
    # shellcheck disable=SC2086
    for n in $(seq 2 ${num_ref}); do
        ref_text_files_str+="text_spk${n} "
        ref_text_names_str+="text_spk${n} "
    done
fi
# shellcheck disable=SC2206
ref_text_files=(${ref_text_files_str// / })
# shellcheck disable=SC2206
ref_text_names=(${ref_text_names_str// / })
# shellcheck disable=SC2206
ref_bc_label_files=(${ref_bc_label_files_str// / })
# shellcheck disable=SC2206
ref_bc_label_names=(${ref_bc_label_names_str// / })

[ -z "${label_train_text}" ] && label_train_text="${data_feats}/org/${train_set}/${ref_bc_label_files[0]}"

label_listdir=data/label_list
if [[ ${category_label} == "merge" ]]; then
    labelword_list="${label_listdir}"/swbd.label.txt
else
    labelword_list="${label_listdir}"/swbd.bin.label.txt
fi
echo "labelword_list: ${labelword_list}"

# Set tag for naming of model directory
if [ -z "${bc_tag}" ]; then
    if [ -n "${bc_config}" ]; then
        bc_tag="$(basename "${bc_config}" .yaml)_${feats_type}"
    else
        bc_tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${bc_args}" ]; then
        bc_tag+="$(echo "${bc_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${bc_stats_dir}" ]; then
    if [ ${only_encoder} = true ]; then
        bc_stats_dir="${expdir}/bc_stats_onlyencoder_${feats_type}"
    else
        bc_stats_dir="${expdir}/bc_stats_${feats_type}"
    fi

    if [ "${lang}" != "noinfo" ]; then
        bc_stats_dir+="_${lang}"
    fi
fi
# The directory used for training commands
if [ -z "${bc_exp}" ]; then
    bc_exp="${expdir}/${bc_tag}"
fi

if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    inference_tag+="_bcp_model_$(echo "${inference_bc_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

if "${skip_data_prep}"; then
    skip_stages+="1 2 3 4 "
fi
if "${skip_train}"; then
    skip_stages+="3 4 5 6 "
fi
if "${skip_eval}"; then
    skip_stages+="7 8 "
fi
skip_stages=$(echo "${skip_stages}" | tr ' ' '\n' | sort -nu | tr '\n' ' ')
log "Skipped stages: ${skip_stages}"

# ========================== Main stages start from here. ==========================

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]] ]]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
    # shellcheck disable=SC2086
    local/data.sh ${local_data_opts}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && ! [[ " ${skip_stages} " =~ [[:space:]]2[[:space:]] ]]; then
    if "${skip_train}"; then
        if "${eval_valid_set}"; then
            _dsets="${valid_set} ${test_sets}"
        else
            _dsets="${test_sets}"
        fi
    else
        _dsets="${train_set} ${valid_set} ${test_sets}"
    fi
    if [ "${feats_type}" = raw ]; then
        log "Stage 2: Format wav.scp: data/ -> ${data_feats}"

        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        for dset in ${_dsets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
            # shellcheck disable=SC2086
            rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}

            # Copy reference text files if there is more than 1 reference
            if [ ${#ref_text_files[@]} -gt 1 ]; then
                echo "check ref_text_files"
                # shellcheck disable=SC2068,SC2086
                for ref_txt in ${ref_text_files[@]}; do
                    [ -f data/${dset}/${ref_txt} ] && cp data/${dset}/${ref_txt} ${data_feats}${_suf}/${dset}
                done
            else
                echo "check not ref_text_files"
            fi

            _opts=
            if [ -e data/"${dset}"/segments ]; then
                # "segments" is used for splitting wav files which are written in "wav".scp
                # into utterances. The file format of segments:
                #   <segment_id> <record_id> <start_time> <end_time>
                #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                # Where the time is written in seconds.
                _opts+="--segments data/${dset}/segments "
            fi
            # shellcheck disable=SC2086,SC2154
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                --multi-columns-input "${multi_columns_input_wav_scp}" \
                --multi-columns-output "${multi_columns_output_wav_scp}" \
                "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

            echo "${feats_type}" >"${data_feats}${_suf}/${dset}/feats_type"
            if "${multi_columns_output_wav_scp}"; then
                echo "multi_${audio_format}" >"${data_feats}${_suf}/${dset}/audio_format"
            else
                echo "${audio_format}" >"${data_feats}${_suf}/${dset}/audio_format"
            fi
        done

    elif [ "${feats_type}" = raw_copy ]; then
        # If you guaranteed that the data already satisfy the raw format, you can skip format_wav_scp.py for reduce the overhead
        for dset in ${_dsets}; do
            if [ -e "data/${dset}/segments" ]; then
                log "Error: data/${dset}/segments is existing. Please use --feats_type raw"
                exit 1
            fi
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"

                if [ -e "data/${dset}/utt2dur" ]; then
                    _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
                    # shellcheck disable=SC2086
                    awk <data/${dset}/utt2dur '{ print $1, int($2*'${_fs}'); }' >"${data_feats}${_suf}/${dset}"/utt2num_samples

                elif [ -e "data/${dset}/utt2num_samples" ]; then
                    cp "data/${dset}/utt2num_samples" "${data_feats}${_suf}/${dset}"/utt2num_samples

                else
                    log "Error: data/${dset}/utt2dur or data/${dset}/utt2num_samples must be existing for train_set and valid_set. Please use --feats_type raw. If you'd like to perform this script for evaluation, please give --skip_train true"
                    exit 1
                fi
            fi

            # Copy reference text files if there is more than 1 reference
            if [ ${#ref_text_files[@]} -gt 1 ]; then
                # shellcheck disable=SC2068,SC2086
                for ref_txt in ${ref_text_files[@]}; do
                    [ -f data/${dset}/${ref_txt} ] && cp data/${dset}/${ref_txt} ${data_feats}${_suf}/${dset}
                done
            fi

            echo "raw" >"${data_feats}${_suf}/${dset}/feats_type"
            if "${multi_columns_input_wav_scp}"; then
                echo "multi_${audio_format}" >"${data_feats}${_suf}/${dset}/audio_format"
            else
                echo "${audio_format}" >"${data_feats}${_suf}/${dset}/audio_format"
            fi
        done

    elif [ "${feats_type}" = fbank_pitch ]; then
        log "[Require Kaldi] Stage 2: ${feats_type} extract: data/ -> ${data_feats}"

        for dset in ${_dsets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            # 1. Copy datadir
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"

            # Copy reference text files if there is more than 1 reference
            if [ ${#ref_text_files[@]} -gt 1 ]; then
                # shellcheck disable=SC2068,SC2086
                for ref_txt in ${ref_text_files[@]}; do
                    [ -f data/${dset}/${ref_txt} ] && cp data/${dset}/${ref_txt} ${data_feats}${_suf}/${dset}
                done
            fi

            # 2. Feature extract
            _nj=$(min "${nj}" "$(wc <"${data_feats}${_suf}/${dset}/utt2spk" -l)")
            steps/make_fbank_pitch.sh --nj "${_nj}" --cmd "${train_cmd}" "${data_feats}${_suf}/${dset}"
            utils/fix_data_dir.sh "${data_feats}${_suf}/${dset}"

            # 3. Derive the the frame length and feature dimension
            scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                "${data_feats}${_suf}/${dset}/feats.scp" "${data_feats}${_suf}/${dset}/feats_shape"

            # 4. Write feats_dim
            # shellcheck disable=SC2086
            head -n 1 "${data_feats}${_suf}/${dset}/feats_shape" | awk '{ print $2 }' |
                cut -d, -f2 >${data_feats}${_suf}/${dset}/feats_dim

            # 5. Write feats_type
            echo "${feats_type}" >"${data_feats}${_suf}/${dset}/feats_type"
        done

    elif [ "${feats_type}" = fbank ]; then
        log "Stage 2: ${feats_type} extract: data/ -> ${data_feats}"
        log "${feats_type} is not supported yet."
        exit 1

    elif [ "${feats_type}" = extracted ]; then
        log "Stage 2: ${feats_type} extract: data/ -> ${data_feats}"
        # Assumming you don't have wav.scp, but feats.scp is created by local/data.sh instead.

        for dset in ${_dsets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            # Generate dummy wav.scp to avoid error by copy_data_dir.sh
            if [ ! -f data/"${dset}"/wav.scp ]; then
                if [ ! -f data/"${dset}"/segments ]; then
                    awk <data/"${dset}"/feats.scp ' { print($1,"<DUMMY>") }' >data/"${dset}"/wav.scp
                else
                    awk <data/"${dset}"/segments ' { print($2,"<DUMMY>") }' >data/"${dset}"/wav.scp
                fi
            fi
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"

            # Copy reference text files if there is more than 1 reference
            # shellcheck disable=SC2068,SC2086
            if [ ${#ref_text_files[@]} -gt 1 ]; then
                for ref_txt in ${ref_text_files[@]}; do
                    [ -f data/${dset}/${ref_txt} ] && cp data/${dset}/${ref_txt} ${data_feats}${_suf}/${dset}
                done
            fi

            # Derive the the frame length and feature dimension
            _nj=$(min "${nj}" "$(wc <"${data_feats}${_suf}/${dset}/utt2spk" -l)")
            scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                "${data_feats}${_suf}/${dset}/feats.scp" "${data_feats}${_suf}/${dset}/feats_shape"

            pyscripts/feats/feat-to-shape.py "scp:head -n 1 ${data_feats}${_suf}/${dset}/feats.scp |" - |
                awk '{ print $2 }' | cut -d, -f2 >"${data_feats}${_suf}/${dset}/feats_dim"

            echo "${feats_type}" >"${data_feats}${_suf}/${dset}/feats_type"
        done

    else
        log "Error: not supported: --feats_type ${feats_type}"
        exit 2
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! [[ " ${skip_stages} " =~ [[:space:]]3[[:space:]] ]]; then
    log "Stage 3: Remove long/short data: ${data_feats}/org -> ${data_feats}"

    # NOTE(kamo): Not applying to test_sets to keep original data
    for dset in "${train_set}" "${valid_set}"; do

        # Copy data dir
        utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
        cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

        # shellcheck disable=SC2086
        # Remove short utterances
        _feats_type="$(<${data_feats}/${dset}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            awk <"${data_feats}/org/${dset}/utt2num_samples" -v min_length="${_min_length}" -v max_length="${_max_length}" \
                '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                >"${data_feats}/${dset}/utt2num_samples"
            utils/filter_scp.pl <"${data_feats}/org/${dset}/wav.scp" "${data_feats}/${dset}/utt2num_samples" \
                >"${data_feats}/${dset}/wav.scp"
        else
            # Get frame shift in ms from conf/fbank.conf
            _frame_shift=
            if [ -f conf/fbank.conf ] && [ "$(grep <conf/fbank.conf -c frame-shift)" -gt 0 ]; then
                # Assume using conf/fbank.conf for feature extraction
                _frame_shift="$(grep <conf/fbank.conf frame-shift | sed -e 's/[-a-z =]*\([0-9]*\)/\1/g')"
            fi
            if [ -z "${_frame_shift}" ]; then
                # If not existing, use the default number in Kaldi (=10ms).
                # If you are using different number, you have to change the following value manually.
                _frame_shift=10
            fi

            _min_length=$(python3 -c "print(int(${min_wav_duration} / ${_frame_shift} * 1000))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} / ${_frame_shift} * 1000))")

            cp "${data_feats}/org/${dset}/feats_dim" "${data_feats}/${dset}/feats_dim"
            awk <"${data_feats}/org/${dset}/feats_shape" -F, ' { print $1 } ' |
                awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length) print $0; }' \
                    >"${data_feats}/${dset}/feats_shape"
            utils/filter_scp.pl <"${data_feats}/org/${dset}/feats.scp" "${data_feats}/${dset}/feats_shape" \
                >"${data_feats}/${dset}/feats.scp"
        fi

        # Remove empty text
        # shellcheck disable=SC2068
        for ref_txt in ${ref_text_files[@]}; do
            awk <"${data_feats}/org/${dset}/${ref_txt}" ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/${ref_txt}"
        done

        # fix_data_dir.sh leaves only utts which exist in all files
        utils/fix_data_dir.sh \
            ${ref_text_files_str:+--utt_extra_files "${ref_text_files_str}"} \
            "${data_feats}/${dset}"
    done

    if [ -n "${post_process_local_data_opts}" ]; then
        # shellcheck disable=SC2086
        # Do any additional local data post-processing here
        local/data.sh ${post_process_local_data_opts} --asr_data_dir "${data_feats}/${train_set}"
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && ! [[ " ${skip_stages} " =~ [[:space:]]4[[:space:]] ]]; then
    # Create label-list for backchannel classification
    log "Stage 4: Generate backchannel label list with expanded label(${category_label})"

    ${python} -m bcp.utils.get_bc_labels \
        --dataset ${dataset_name} \
        --category_label ${category_label} \
        --output "${labelword_list}"
fi
# ========================== Data preparation is done here. ==========================

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && ! [[ " ${skip_stages} " =~ [[:space:]]5[[:space:]] ]]; then
    _bcp_train_dir="${data_feats}/${train_set}"
    _bcp_valid_dir="${data_feats}/${valid_set}"
    log "Stage 5: BCP collect stats: train_set=${_bcp_train_dir}, valid_set=${_bcp_valid_dir}"

    _opts=
    if [ -n "${bc_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.bcp_train --print_config --optim adam
        _opts+="--config ${bc_config} "
    fi

    _feats_type="$(<${_bcp_train_dir}/feats_type)"
    _audio_format="$(cat ${_bcp_train_dir}/audio_format 2>/dev/null || echo ${audio_format})"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        if [[ "${_audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi
        _opts+="--frontend_conf fs=${fs} "

        for fronted_opt in ${fronted_opts}; do
            _opts+="--frontend_conf ${fronted_opt} "
        done
    else
        _scp=feats.scp
        _type=kaldi_ark
        _input_size="$(<${_bcp_train_dir}/feats_dim)"
        _opts+="--input_size=${_input_size} "
    fi

    # 1. Split the key file
    _logdir="${bc_stats_dir}/logdir"
    mkdir -p "${_logdir}"

    # Get the minimum number among ${nj} and the number lines of input files
    _nj=$(min "${nj}" "$(wc <${_bcp_train_dir}/${_scp} -l)" "$(wc <${_bcp_valid_dir}/${_scp} -l)")

    key_file="${_bcp_train_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${_bcp_valid_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/valid.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Generate run.sh
    log "Generate '${bc_stats_dir}/run.sh'. You can resume the process from stage 5 using this script"
    mkdir -p "${bc_stats_dir}"
    echo "${run_args} --stage 5 \"\$@\"; exit \$?" >"${bc_stats_dir}/run.sh"
    chmod +x "${bc_stats_dir}/run.sh"

    # 3. Submit jobs
    log "BC collect-stats started... log: '${_logdir}/stats.*.log'"

    # add option to use preprocessor
    _opts+="--preprocessor classification "
    _opts+="--preprocessor_conf classification_name=classification --preprocessor_conf classification_list=${labelword_list} --preprocessor_conf category=${category_label} "

    # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
    #       but it's used only for deciding the sample ids.

    _opts+="--train_data_path_and_name_and_type ${_bcp_train_dir}/${_scp},speech,${_type} "
    _opts+="--valid_data_path_and_name_and_type ${_bcp_valid_dir}/${_scp},speech,${_type} "

    # shellcheck disable=SC2068
    for i in ${!ref_bc_label_files[@]}; do
        _opts+="--train_data_path_and_name_and_type ${_bcp_train_dir}/${ref_bc_label_files[$i]},${ref_bc_label_names[$i]},text "
        _opts+="--valid_data_path_and_name_and_type ${_bcp_valid_dir}/${ref_bc_label_files[$i]},${ref_bc_label_names[$i]},text "
    done

    if [ ${only_encoder} = false ]; then
        # shellcheck disable=SC2068
        for i in ${!ref_text_files[@]}; do
            _opts+="--train_data_path_and_name_and_type ${_bcp_train_dir}/${ref_text_files[$i]},${ref_text_names[$i]},text "
            _opts+="--valid_data_path_and_name_and_type ${_bcp_valid_dir}/${ref_text_files[$i]},${ref_text_names[$i]},text "
        done
        _opts+="--bpemodel ${bpemodel} "
        _opts+="--token_type ${token_type} "
        _opts+="--token_list ${token_list} "
    fi

    # shellcheck disable=SC2046,SC2086
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        ${python} -m bcp.bin.${bc_task}_train \
        --collect_stats true \
        --use_preprocessor true \
        --non_linguistic_symbols "${nlsyms_txt}" \
        --cleaner "${cleaner}" \
        --g2p "${g2p}" \
        --train_shape_file "${_logdir}/train.JOB.scp" \
        --valid_shape_file "${_logdir}/valid.JOB.scp" \
        --output_dir "${_logdir}/stats.JOB" \
        ${_opts} ${bc_args} || {
        cat $(grep -l -i error "${_logdir}"/stats.*.log)
        exit 1
    }

    # 4. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    if [ "${feats_normalize}" != global_mvn ]; then
        # Skip summerizaing stats if not using global MVN
        _opts+="--skip_sum_stats"
    fi
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${bc_stats_dir}"

    # Append the num-tokens at the last dimensions. This is used for batch-bins count
    # shellcheck disable=SC2068
    for ref_txt in ${ref_text_names[@]}; do
        # shellcheck disable=SC2086
        awk <"${bc_stats_dir}/train/${ref_txt}_shape" -v N="$(wc <${token_list} -l)" '{ print $0 "," N }' \
            >"${bc_stats_dir}/train/${ref_txt}_shape.${token_type}"

        # shellcheck disable=SC2086
        awk <"${bc_stats_dir}/valid/${ref_txt}_shape" -v N="$(wc <${token_list} -l)" '{ print $0 "," N }' \
            >"${bc_stats_dir}/valid/${ref_txt}_shape.${token_type}"
    done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ] && ! [[ " ${skip_stages} " =~ [[:space:]]6[[:space:]] ]]; then
    _bcp_train_dir="${data_feats}/${train_set}"
    _bcp_valid_dir="${data_feats}/${valid_set}"
    log "Stage 6: Backchannel Prediction Model Training: train_set=${_bcp_train_dir}, valid_set=${_bcp_valid_dir}"

    _opts=
    if [ -n "${bc_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
        _opts+="--config ${bc_config} "
    fi

    _feats_type="$(<${_bcp_train_dir}/feats_type)"
    _audio_format="$(cat ${_bcp_train_dir}/audio_format 2>/dev/null || echo ${audio_format})"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        # "sound" supports "wav", "flac", etc.
        if [[ "${_audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        elif [[ "${_audio_format}" == *multi* ]]; then
            _type=multi_columns_sound
        else
            _type=sound
        fi
        _fold_length="$((bc_speech_fold_length * 100))"
        _opts+="--frontend_conf fs=${fs} "

        for fronted_opt in ${fronted_opts}; do
            _opts+="--frontend_conf ${fronted_opt} "
        done
    else
        _scp=feats.scp
        _type=kaldi_ark
        _fold_length="${bc_speech_fold_length}"
        _input_size="$(<${_bcp_train_dir}/feats_dim)"
        _opts+="--input_size=${_input_size} "

    fi

    _preopts="--preprocessor classification "
    _preopts+="--preprocessor_conf classification_name=classification --preprocessor_conf classification_list=${labelword_list} --preprocessor_conf category=${category_label} "

    if [ "${feats_normalize}" = global_mvn ]; then
        # Default normalization is utterance_mvn and changes to global_mvn
        _opts+="--normalize=global_mvn --normalize_conf stats_file=${bc_stats_dir}/train/feats_stats.npz "
    fi

    _opts+="--train_data_path_and_name_and_type ${_bcp_train_dir}/${_scp},speech,${_type} "
    _opts+="--train_shape_file ${bc_stats_dir}/train/speech_shape "

    _opts+="--fold_length ${bc_label_fold_length} "
    _opts+="--train_data_path_and_name_and_type ${_bcp_train_dir}/bc_label,classification,text "
    _opts+="--train_shape_file ${bc_stats_dir}/train/classification_shape "

    _opts+="--valid_data_path_and_name_and_type ${_bcp_valid_dir}/bc_label,classification,text "
    _opts+="--valid_shape_file ${bc_stats_dir}/valid/classification_shape "

    if [ "${only_encoder}" = false ]; then
        _opts+="--fold_length ${bc_text_fold_length} "
        _opts+="--train_data_path_and_name_and_type ${_bcp_train_dir}/text,text,text "
        _opts+="--train_shape_file ${bc_stats_dir}/train/text_shape.${token_type} "
        _opts+="--valid_data_path_and_name_and_type ${_bcp_valid_dir}/text,text,text "
        _opts+="--valid_shape_file ${bc_stats_dir}/valid/text_shape.${token_type} "
        _opts+="--bpemodel ${bpemodel} "
        _opts+="--token_type ${token_type} "
        _opts+="--token_list ${token_list} "
    fi

    if [ "${use_wandb}" = true ]; then
        _opts+="--use_wandb true "
        _opts+="--wandb_project ${wandb_project} "
        _opts+="--wandb_name ${wandb_name} "
    fi

    log "Generate '${bc_exp}/run.sh'. You can resume the process from stage 6 using this script"
    mkdir -p "${bc_exp}"
    echo "${run_args} --stage 6 \"\$@\"; exit \$?" >"${bc_exp}/run.sh"
    chmod +x "${bc_exp}/run.sh"

    # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
    log "ASR training started... log: '${bc_exp}/train.log'"
    # shellcheck disable=SC2154
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &>/dev/null; then
        # SGE can't include "/" in a job name
        # shellcheck disable=SC2086
        jobname="$(basename ${bc_exp})"
    else
        jobname="${bc_exp}/train.log"
    fi

    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log "${bc_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${bc_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m bcp.bin.${bc_task}_train \
        --use_preprocessor true \
        ${_preopts} \
        --non_linguistic_symbols "${nlsyms_txt}" \
        --cleaner "${cleaner}" \
        --g2p "${g2p}" \
        --valid_data_path_and_name_and_type "${_bcp_valid_dir}/${_scp},speech,${_type}" \
        --valid_shape_file "${bc_stats_dir}/valid/speech_shape" \
        --resume true \
        ${pretrained_model:+--init_param $pretrained_model} \
        --ignore_init_mismatch ${ignore_init_mismatch} \
        --fold_length "${_fold_length}" \
        --output_dir "${bc_exp}" \
        ${_opts} ${bc_args}
fi

# if [ -n "${download_model}" ]; then
#     log "Use ${download_model} for decoding and evaluation"
#     bc_exp="${expdir}/${download_model}"
#     mkdir -p "${bc_exp}"

#     # If the model already exists, you can skip downloading
#     espnet_model_zoo_download --unpack true "${download_model}" >"${bc_exp}/config.txt"

#     # Get the path of each file
#     _bcp_model_file=$(sed <"${bc_exp}/config.txt" -e "s/.*'bcp_model_file': '\([^']*\)'.*$/\1/")
#     _bcp_train_config=$(sed <"${bc_exp}/config.txt" -e "s/.*'bcp_train_config': '\([^']*\)'.*$/\1/")

#     # Create symbolic links
#     ln -sf "${_bcp_model_file}" "${bc_exp}"
#     ln -sf "${_bcp_train_config}" "${bc_exp}"
#     inference_bc_model=$(basename "${_bcp_model_file}")
# fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ] && ! [[ " ${skip_stages} " =~ [[:space:]]7[[:space:]] ]]; then
    log "Stage 7: Predicting: training_dir=${bc_exp}"

    if ${gpu_inference}; then
        _cmd="${cuda_cmd}"
        _ngpu=1
    else
        # shellcheck disable=SC2154
        _cmd="${decode_cmd}"
        _ngpu=0
    fi

    _opts=
    if [ -n "${inference_config}" ]; then
        _opts+="--config ${inference_config} "
    fi

    # 2. Generate run.sh
    log "Generate '${bc_exp}/${inference_tag}/run.sh'. You can resume the process from stage 7 using this script"
    mkdir -p "${bc_exp}/${inference_tag}"
    echo "${run_args} --stage 7 \"\$@\"; exit \$?" >"${bc_exp}/${inference_tag}/run.sh"
    chmod +x "${bc_exp}/${inference_tag}/run.sh"

    inference_bin_tag=""
    if "${use_streaming}"; then
        inference_bin_tag="_streaming"
        _opts+="--sim_chunk_length 512 "
    fi

    if "${eval_valid_set}"; then
        _dsets="org/${valid_set} ${test_sets}"
    else
        _dsets="${test_sets}"
    fi
    for dset in ${_dsets}; do
        _data="${data_feats}/${dset}"
        _dir="${bc_exp}/${inference_tag}/${dset}"
        _logdir="${_dir}/logdir"
        mkdir -p "${_logdir}"

        # shellcheck disable=SC2086
        _feats_type="$(<${_data}/feats_type)"
        # shellcheck disable=SC2086
        _audio_format="$(cat ${_data}/audio_format 2>/dev/null || echo ${audio_format})"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            elif [[ "${_audio_format}" == *multi* ]]; then
                _type=multi_columns_sound
            else
                _type=sound
            fi
        else
            _scp=feats.scp
            _type=kaldi_ark
        fi

        # 1. Split the key file
        key_file=${_data}/${_scp}
        split_scps=""
        # shellcheck disable=SC2086
        _nj=$(min "${inference_nj}" "$(wc <${key_file} -l)")

        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit decoding jobs
        log "Predicting started... log: '${_logdir}/bcp_inference.*.log'"
        rm -f "${_logdir}/*.log"
        # shellcheck disable=SC2046,SC2086
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/bcp_inference.JOB.log \
            ${python} -m bcp.bin.${bc_task}_inference${inference_bin_tag} \
            --batch_size ${batch_size} \
            --ngpu "${_ngpu}" \
            --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
            --key_file "${_logdir}"/keys.JOB.scp \
            --asr_train_config ${bc_exp}/config.yaml \
            --asr_model_file ${bc_exp}/${inference_bc_model} \
            --output_dir ${_logdir}/output.JOB \
            ${_opts} ${inference_args} || {
            cat $(grep -l -i error "${_logdir}"/bcp_inference.*.log)
            exit 1
        }

        # 3. Concatenates the output files from each jobs
        # shellcheck disable=SC2068
        for f in bc bc_int bc_probs; do
            if [ -f "${_logdir}/output.1/${f}" ]; then
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/${f}"
                done | sort -k1 >"${_dir}/${f}"
            fi
        done

        # remove decoding
        # 4. Concatenates the output files from each jobs for multitask
        # shellcheck disable=SC2068
        for ref_txt in ${ref_text_files[@]}; do
            suffix=$(echo ${ref_txt} | sed 's/text//')
            for f in token token_int score text; do
                if [ -f "${_logdir}/output.1/1best_recog/${f}${suffix}" ]; then
                    for i in $(seq "${_nj}"); do
                        cat "${_logdir}/output.${i}/1best_recog/${f}${suffix}"
                    done | sort -k1 >"${_dir}/${f}${suffix}"
                fi
            done
        done
    done
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ] && ! [[ " ${skip_stages} " =~ [[:space:]]8[[:space:]] ]]; then
    log "Stage 8: Scoring for backchannel prediction"

    if "${eval_valid_set}"; then
        _dsets="org/${valid_set} ${test_sets}"
    else
        _dsets="${test_sets}"
    fi

    
    for dset in ${_dsets}; do
        _data="${data_feats}/${dset}"
        _dir="${bc_exp}/${inference_tag}/${dset}"

        if [ ${block_mapping} = true ]; then
            ${python} -m bcp.eval.bc_score \
                --exp_root "./dump_and_exp" \
                --dataset_name "${dataset_name}" \
                --dataset_tag "${dataset_type}" \
                --bc_tag "${bc_tag}" \
                --inference_tag "${inference_tag}" \
                --category_label "${category_label}" \
                --test_name "${dset}" \
                --block_mapping
        else
            ${python} -m bcp.eval.bc_score \
                --exp_root "./dump_and_exp" \
                --dataset_name "${dataset_name}" \
                --dataset_tag "${dataset_type}" \
                --bc_tag "${bc_tag}" \
                --inference_tag "${inference_tag}" \
                --category_label "${category_label}" \
                --test_name "${dset}"
        fi
    done
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ] && ! [[ " ${skip_stages} " =~ [[:space:]]8[[:space:]] ]]; then
    log "Stage 9: ASR Evaluation for backchannel prediction"

    if "${eval_valid_set}"; then
        _dsets="org/${valid_set} ${test_sets}"
    else
        _dsets="${test_sets}"
    fi
    for dset in ${_dsets}; do
        _data="${data_feats}/${dset}"
        _dir="${bc_exp}/${inference_tag}/${dset}"

        for _tok_type in "char" "word" "bpe"; do
            [ "${_tok_type}" = bpe ] && [ ! -f "${bpemodel}" ] && continue

            _opts="--token_type ${_tok_type} "
            if [ "${_tok_type}" = "char" ] || [ "${_tok_type}" = "word" ]; then
                _type="${_tok_type:0:1}er"
                _opts+="--non_linguistic_symbols ${nlsyms_txt} "
                _opts+="--remove_non_linguistic_symbols true "

            elif [ "${_tok_type}" = "bpe" ]; then
                _type="ter"
                _opts+="--bpemodel ${bpemodel} "

            else
                log "Error: unsupported token type ${_tok_type}"
            fi

            _scoredir="${_dir}/score_${_type}"
            mkdir -p "${_scoredir}"

            # shellcheck disable=SC2068
            for ref_txt in ${ref_text_files[@]}; do
                # Note(simpleoier): to get the suffix after text, e.g. "text_spk1" -> "_spk1"
                suffix=$(echo ${ref_txt} | sed 's/text//')

                # Tokenize text to ${_tok_type} level
                paste \
                    <(
                        ${python} <"${_data}/${ref_txt}" -m espnet2.bin.tokenize_text \
                            -f 2- --input - --output - \
                            --cleaner "${cleaner}" \
                            ${_opts}
                    ) \
                    <(awk <"${_data}/utt2spk" '{ print "(" $2 "-" $1 ")" }') \
                    >"${_scoredir}/ref${suffix:-${suffix}}.trn"

                # NOTE(kamo): Don't use cleaner for hyp
                paste \
                    <(
                        ${python} <"${_dir}/${ref_txt}" -m espnet2.bin.tokenize_text \
                            -f 2- --input - --output - \
                            ${_opts} \
                            --cleaner "${hyp_cleaner}"
                    ) \
                    <(awk <"${_data}/utt2spk" '{ print "(" $2 "-" $1 ")" }') \
                    >"${_scoredir}/hyp${suffix:-${suffix}}.trn"

            done

            # Note(simpleoier): score across all possible permutations
            if [ ${num_ref} -gt 1 ] && [ -n "${suffix}" ]; then
                for i in $(seq ${num_ref}); do
                    for j in $(seq ${num_inf}); do
                        sclite \
                            ${score_opts} \
                            -r "${_scoredir}/ref_spk${i}.trn" trn \
                            -h "${_scoredir}/hyp_spk${j}.trn" trn \
                            -i rm -o all stdout >"${_scoredir}/result_r${i}h${j}.txt"
                    done
                done
                # Generate the oracle permutation hyp.trn and ref.trn
                pyscripts/utils/eval_perm_free_error.py --num-spkrs ${num_ref} \
                    --results-dir ${_scoredir}
            fi

            sclite \
                ${score_opts} \
                -r "${_scoredir}/ref.trn" trn \
                -h "${_scoredir}/hyp.trn" trn \
                -i rm -o all stdout >"${_scoredir}/result.txt"

            log "Write ${_type} result in ${_scoredir}/result.txt"
            grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
        done
    done

    [ -f local/score.sh ] && local/score.sh ${local_score_opts} "${bc_exp}"

    # Show results in Markdown syntax
    scripts/utils/show_asr_result.sh "${bc_exp}" >"${bc_exp}"/RESULTS.md
    cat "${bc_exp}"/RESULTS.md
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
