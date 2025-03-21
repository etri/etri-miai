#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

bc_config=conf/baseline.yaml
bc_tag=multitask.asr_bc
block_mapping=false
category_label=merge
cuda=0
dataset_name="swbd"
dataset_type=swbd.diag.default.pc.beta.512_128_8_16_16
inference_bc_model="valid.macro_f1_scores.ave.pth"
inference_config="conf/decode_asr_streaming.yaml"
only_encoder=false
pretrained=""
start_stage="1"
stop_stage="9"
token_list="./example/token_list/bpe_unigram2000/tokens.txt"
token_type="bpe"
bpe_model="./example/token_list/bpe_unigram2000/bpe.model"
use_wandb=false
wandb_project=
wandb_name=

# 스크립트 시작 시간 기록
start_time=$(date +"%Y-%m-%d %H:%M:%S")

run_args=$(scripts/utils/print_args.sh $0 "$@")

# 명령줄 인자 파싱
while [ $# -gt 0 ]; do
  case "$1" in
  --block_mapping)
    block_mapping=true
    shift
    ;;
  --cuda)
    cuda="$2"
    shift 2
    ;;
  --dataset_name)
    dataset_name="$2"
    shift 2
    ;;
  --bc_tag)
    bc_tag="$2"
    shift 2
    ;;
  --bc_config)
    bc_config="$2"
    shift 2
    ;;
  --pretrained)
    pretrained="$2"
    shift 2
    ;;
  --dataset_type)
    dataset_type="$2"
    shift 2
    ;;
  --start_stage)
    start_stage="$2"
    shift 2
    ;;
  --stop_stage)
    stop_stage="$2"
    shift 2
    ;;
  --inference_config)
    inference_config="$2"
    shift 2
    ;;
  --only_encoder)
    only_encoder=true
    shift
    ;;
  --token_list)
    token_list="$2"
    shift 2
    ;;
  --token_type)
    token_type="$2"
    shift 2
    ;;
  --bpe_model)
    bpe_model="$2"
    shift 2
    ;;
  --inference_bc_model)
    inference_bc_model="$2"
    shift 2
    ;;  
  --category_label)
    category_label="$2"
    shift 2
    ;;
  --use_wandb)
    use_wandb=true
    shift
    ;;
  --wandb_project)
    wandb_project="$2"
    shift 2
    ;;
  --)
    shift
    break
    ;;
  *) break ;;
  esac
done

# 변수 사용 예제
echo "CUDA: $cuda"
echo "Dataset: $dataset_name"
echo "BC Tag: $bc_tag"
echo "BC Config: $bc_config"
echo "Pretrained: $pretrained"
echo "Dataset Type: $dataset_type"
echo "Start Stage: $start_stage"
echo "Stop Stage: $stop_stage"
echo "Inference Config: $inference_config"
echo "Only Encoder: $only_encoder"
echo "Token List: $token_list"
echo "Token Type: $token_type"
echo "BPE Model: $bpe_model"
echo "Inference BC Model: ${inference_bc_model}"
echo "Category Label: ${category_label}"
echo "Block Mapping: ${block_mapping}"
echo "Use Wandb: ${use_wandb}"
if [ "${use_wandb}" = true ]; then
  wandb_name="${dataset_name}_${dataset_type}_${bc_tag}_${category_label}"
  echo "Wandb Project: ${wandb_project}"
  echo "Wandb Name: ${wandb_name}"
fi

export CUDA_VISIBLE_DEVICES="${cuda}"

ngpu=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F "," '{print NF}')
echo "Number of GPUs: ${ngpu}"
fronted_opts=""

if [ -n "${pretrained}" ]; then
  opt_pretrained="${pretrained}:frontend:frontend "
  opt_pretrained+="${pretrained}:normalize:normalize "
  opt_pretrained+="${pretrained}:encoder:encoder "
  # opt_pretrained+="${pretrained}:classifier:classifier" # for the adaption of the classifier
else
  opt_pretrained=""
fi

./bc.sh \
  --stage ${start_stage} --stop_stage ${stop_stage} \
  --use_streaming true \
  --lang en \
  --ngpu ${ngpu} \
  --nj 16 \
  --gpu_inference true \
  --inference_nj 4 \
  --feats_type "raw" \
  --audio_format "wav" \
  --bc_tag "${bc_tag}" \
  --bc_config "${bc_config}" \
  --only_encoder "${only_encoder}" \
  --max_wav_duration 20 \
  --pretrained_model "${opt_pretrained}" \
  --fronted_opts "${fronted_opts}" \
  --category_label "${category_label}" \
  --inference_config "${inference_config}" \
  --token_type "${token_type}" \
  --token_list "${token_list}" \
  --bpemodel "${bpe_model}" \
  --dataset_name ${dataset_name} \
  --dataset_type ${dataset_type} \
  --inference_bc_model "${inference_bc_model}" \
  --block_mapping ${block_mapping} \
  --use_wandb ${use_wandb} \
  --wandb_project "${wandb_project}" \
  --wandb_name "${wandb_name}"

# 스크립트 시작 시간 기록 
end_time=$(date +"%Y-%m-%d %H:%M:%S")

# 성공 시에 돌린 스크립트 저장
echo "Start Time: ${start_time}\nEnd Time: ${end_time}\n\nCompleted Script: ${run_args}"
