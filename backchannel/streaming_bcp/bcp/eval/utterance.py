"""
Data verification functions.
"""

from typing import Dict

from bcp.eval.block_mapping import BlockMapper


class Utterance:

    def __init__(
        self,
        cfg: Dict,
        preds: Dict[str, str],
        golds: Dict[str, str],
        block_mapping=False,
        locator=None,
    ):
        self.cfg = cfg
        self.block_mapping = block_mapping
        self.check_keys(preds, golds)

        self.preds, self.golds = self.check_labels(preds, golds)
        self.balanced_preds, self.balanced_golds, self.filtered_infos = (
            self.extract_balanced_labels(self.preds, self.golds)
        )

        # Dictionary를 key 기준으로 정렬
        self.preds = dict(sorted(self.preds.items()))
        self.golds = dict(sorted(self.golds.items()))
        self.balanced_preds = dict(sorted(self.balanced_preds.items()))
        self.balanced_golds = dict(sorted(self.balanced_golds.items()))

        # Blockmap cfg가 있을 경우, Block Mapping을 수행하여 preds, golds를 재추출
        if self.block_mapping:
            tgt_block_bound_infos = self.get_target_block_bound_infos()
            block_mapper = BlockMapper(locator, tgt_block_bound_infos)

            self.mapping_preds = block_mapper.get_mapping_labels(self.preds)
            self.mapping_golds = block_mapper.get_mapping_labels(self.golds)

            self.mapping_preds = dict(sorted(self.mapping_preds.items()))
            self.mapping_golds = dict(sorted(self.mapping_golds.items()))

            # full 버전만 지원, balanced는 추후 지원 예정

    def check_keys(self, preds: Dict[str, str], golds: Dict[str, str]):
        """
        Check if the keys of the prediction and ground truth are the same.
        """

        for pred_key, gold_key in zip(preds.keys(), golds.keys()):
            if pred_key != gold_key:
                raise KeyError(f"Key 불일치: {pred_key} vs {gold_key}")

    def check_labels(self, preds: Dict[str, str], golds: Dict[str, str]):
        """
        Check if the values of the prediction and ground truth are the same.
        """
        new_preds = {}
        new_golds = {}
        for key in preds.keys():
            pred_val = preds[key]
            gold_val = golds[key]

            if len(pred_val) - 1 == len(gold_val):
                new_preds[key] = pred_val[:-1]
                new_golds[key] = gold_val
            else:
                new_preds[key] = pred_val
                new_golds[key] = gold_val

            new_pred_val = new_preds[key]
            new_gold_val = new_golds[key]

            if len(new_pred_val) != len(new_gold_val):
                raise ValueError(
                    f"Value의 크기 불일치({key}): pred -> {len(pred_val)} vs gold ->{len(gold_val)}"
                )

        return new_preds, new_golds

    def extract_balanced_labels(
        self, preds: Dict[str, str], golds: Dict[str, str], before_nobc: int = 32000
    ):
        def get_subsample(name):
            subsample_map = {
                "linear": 1,
                "conv2d": 4,
                "conv2d6": 6,
                "conv2d8": 8,
                "embed": 1,
                None: 1,
            }
            if name not in subsample_map:
                raise ValueError(f"unknown input_layer: {name}")
            return subsample_map[name]

        if "frontend_conf" in self.cfg and "hop_length" in self.cfg["frontend_conf"]:
            hop_length = self.cfg["frontend_conf"]["hop_length"]
        else:
            hop_length = 128

        if "encoder_conf" in self.cfg:
            if "input_layer" in self.cfg["encoder_conf"]:
                input_layer = self.cfg["encoder_conf"]["input_layer"]
                subsample = get_subsample(input_layer)
            else:
                raise KeyError("encoder_conf.input_layer가 없습니다.")

            if "hop_size" in self.cfg["encoder_conf"]:
                hop_size = self.cfg["encoder_conf"]["hop_size"]
            else:
                raise KeyError("encoder_conf.hop_size가 없습니다.")

        num_hop_samples = hop_length * subsample * hop_size

        # BC 라벨로 지정된 Block의 2초(32000) 전에 근사한 블록을 NOBC 블록으로 추출하고자 함
        nobc_idx = round(before_nobc / num_hop_samples)

        balanced_preds, balanced_golds = {}, {}
        filtered = {}

        for key in preds.keys():
            pred_val = preds[key]
            gold_val = golds[key]

            bc_indices = [i for i, item in enumerate(gold_val) if item != "NOBC"]
            non_bc_indices = [i - nobc_idx for i in bc_indices if i - nobc_idx >= 0]
            indices = sorted(list(set(bc_indices + non_bc_indices)))

            filtered[key] = {
                "num_bc": len(bc_indices),
                "num_nobc": len(non_bc_indices),
                "num_total": len(indices),
                "filtered_bc": len(bc_indices) + len(non_bc_indices) - len(indices),
            }

            if len(indices) > 0:
                balanced_preds[key] = [pred_val[i] for i in indices]
                balanced_golds[key] = [gold_val[i] for i in indices]

        return balanced_preds, balanced_golds, filtered

    def get_preds(self):
        return self.preds

    def get_golds(self):
        return self.golds

    def get_balanced_preds(self):
        return self.balanced_preds

    def get_balanced_golds(self):
        return self.balanced_golds

    def get_mapping_preds(self):
        return self.mapping_preds

    def get_mapping_golds(self):
        return self.mapping_golds

    def get_target_block_bound_infos(self):
        block_infos = {
            "stft": {"win_len": 320, "hop_len": 128, "center": True},
            "block": {
                "block_size": 40,
                "previous_size": 8,
                "current_size": 16,
                "lookahead_size": 16,
                "hop_size": 16,
            },
        }

        # cfg에 frontend_conf가 있을 경우, n_fft, hop_len을 업데이트
        if "frontend_conf" in self.cfg:
            if "n_fft" in self.cfg["frontend_conf"]:
                block_infos["stft"]["n_fft"] = self.cfg["frontend_conf"]["n_fft"]
            if "win_length" in self.cfg["frontend_conf"]:
                block_infos["stft"]["win_len"] = self.cfg["frontend_conf"]["win_length"]
            if "hop_length" in self.cfg["frontend_conf"]:
                block_infos["stft"]["hop_len"] = self.cfg["frontend_conf"]["hop_length"]

        # cfg에 encoder_conf가 있을 경우, block_size, hop_size를 업데이트
        if "encoder_conf" in self.cfg:
            if "block_size" in self.cfg["encoder_conf"]:

                block_size = self.cfg["encoder_conf"]["block_size"]

                block_infos["block"]["current_size"] = self.cfg["encoder_conf"][
                    "hop_size"
                ]
                block_infos["block"]["lookahead_size"] = self.cfg["encoder_conf"][
                    "look_ahead"
                ]
                block_infos["block"]["previous_size"] = block_size - (
                    self.cfg["encoder_conf"]["hop_size"]
                    + self.cfg["encoder_conf"]["look_ahead"]
                )
                block_infos["block"]["block_size"] = block_size
                block_infos["block"]["hop_size"] = block_infos["block"]["current_size"]

        return block_infos

    def get_total_filtered_infos(self):
        infos = {
            "num_bc": 0,
            "num_nobc": 0,
            "num_total": 0,
            "filtered_bc": 0,
        }

        for v in self.filtered_infos.values():
            infos["num_bc"] += v["num_bc"]
            infos["num_nobc"] += v["num_nobc"]
            infos["num_total"] += v["num_total"]
            infos["filtered_bc"] += v["filtered_bc"]

        return infos
