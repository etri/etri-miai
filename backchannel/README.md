
## Backchannel Repository
This repository is for backchannel using Switchboard corpus.
- Backchannel corpora
  - Database: ([swbd](https://catalog.ldc.upenn.edu/LDC97S62))
  - Backchannel Label: ([backchannel](https://github.com/phiresky/backchannel-prediction/blob/master/data/utterance_is_backchannel.json))

## Usage
- Install ESPnet toolkit ([doc](https://espnet.github.io/espnet/installation.html))
  1. Git clone ESPnet
  2. Put compiled Kaldi under espnet/tools
  3. Setup Python environment
  4. Install ESPnet
- Run backchannel script
  1. Set ESPnet path (Modify {MAIN_ROOT} in path.sh (line 1))
  2. Set SWBD corpus path (Modify {datadir} in run.sh (line 47))
  3. Run run.sh

## Experiments
### Enviroments
- date: `Wed Nov  30 00:22:13 EST 2022`
- python version: `3.8.13 (default, Oct 21 2022, 23:50:54)  [GCC 11.2.0]`
- espnet version: `espnet 202209`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.12.1`

- Model files (archived to model.tar.gz by `$ pack_model.sh`)
  - training config file: `conf/tuning/train_pytorch_transformer.yaml`
  - decoding config file: `conf/decode.yaml`
  - cmvn file: `data/train/cmvn.ark`
  - e2e file: `exp/train_pytorch_train_pytorch_transformer_specaug/results/model.val5.avg.best`
  - e2e JSON file: `exp/train_pytorch_train_pytorch_transformer_specaug/results/model.json`
  - dict file: `data/lang_char`

### CER
```
exp/train_pytorch_train_pytorch_transformer_specaug/decode_eval_model.val5.avg.best_decode_lm/result.txt
|  SPKR                 |  # Snt    # Wrd  |  Corr       Sub      Del       Ins       Err      S.Err  |
|  Sum/Avg              | 33583     387055 |  87.0       7.8      5.2       2.5      15.5       38.5  |
```
### WER
```
exp/train_pytorch_train_pytorch_transformer_specaug/decode_eval_model.val5.avg.best_decode_lm/result.wrd.txt
|  SPKR                  |  # Snt    # Wrd   |  Corr       Sub      Del       Ins        Err      S.Err   |
|  Sum/Avg               | 33583     279667  |  87.1       9.9      3.0       2.6       15.5       38.4   |
```
