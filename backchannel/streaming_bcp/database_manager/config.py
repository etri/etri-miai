from typing import List
import yaml
from pathlib import Path

from utils import singleton

SCRIPT = Path(__file__)


class Config:
    name = None
    db_root = None
    audio = None
    stft = None
    conv = None
    block = None
    policy = None

    @classmethod
    def load(cls, config_name, verbose=False):
        # 설정 이름.
        # 후에, espnet dataset 이름으로 활용함
        cls.name = config_name

        config_path = SCRIPT.parent / "config" / f"{cls.name}.yaml"
        assert config_path.exists(), f"설정 파일 {config_path}가 없습니다."

        with open(config_path, "r") as config_f:
            config = yaml.load(config_f, yaml.FullLoader)

        # 설정 파일 key 확인
        for key in ["DB_root", "audio", "stft", "conv", "block", "policy"]:
            assert (
                key in config.keys()
            ), f"설정 파일은 반드시 '{key}' 항목을 포함해야 합니다."

        # db root 설정
        cls.db_root = Path(config["DB_root"])
        assert cls.db_root.exists(), f"DB root {cls.db_root}가 존재하지 않습니다."

        cls.audio = config["audio"]
        for key in ["sample_rate"]:
            assert (
                key in cls.audio.keys()
            ), f"audio 항목은 반드시 {key} 항목을 포함해야 합니다."

        # stft/conv/block => block labeling 관련 설정
        cls.stft = config["stft"]
        for key in ["n_fft", "hop_len"]:
            assert (
                key in cls.stft.keys()
            ), f"stft 항목은 반드시 {key} 항목을 포함해야 합니다."

        cls.conv = config["conv"]
        for key in ["kernel", "stride"]:
            assert (
                key in cls.conv.keys()
            ), f"conv 항목은 반드시 {key} 항목을 포함해야 합니다."

        cls.block = config["block"]
        for key in [
            "block_size",
            "previous_size",
            "current_size",
            "lookahead_size",
            "hop_size",
        ]:
            assert (
                key in cls.block.keys()
            ), f"block 항목은 반드시 {key} 항목을 포함해야 합니다."

        # traning utterance 생성 관련 policy
        cls.policy = config["policy"]
        for key in ["filter", "end", "label", "split"]:
            assert (
                key in cls.policy.keys()
            ), f"policy 항목은 반드시 {key} 항목을 포함해야 합니다."

        if verbose:
            print(f"============ Configuration =============")
            print(f"- name              : {cls.name}")
            print(f"- audio --------------------------------")
            print(f"    * sample rate   : {cls.audio['sample_rate']}")
            print(f"- stft --------------------------------")
            print(f"    * n fft         : {cls.stft['n_fft']}")
            print(f"    * hop len       : {cls.stft['hop_len']}")
            print(f"- conv --------------------------------")
            print(f"    * kernel        : {cls.conv['kernel']}")
            print(f"    * stride        : {cls.conv['stride']}")
            print(f"- block --------------------------------")
            print(f"    * block size    : {cls.block['block_size']}")
            print(f"    * previous sizes: {cls.block['previous_size']}")
            print(f"    * current size  : {cls.block['current_size']}")
            print(f"    * lookahead size: {cls.block['lookahead_size']}")
            print(f"    * hop size      : {cls.block['hop_size']}")
            print(f"- policy --------------------------------")
            print(f"    * filter        : {cls.policy['filter']}")
            print(f"    * end           : {cls.policy['end']}")
            print(f"    * label         : {cls.policy['label']}")
            print(f"    * split         : {cls.policy['split']}")
            print()


# @singleton
# class Config:
#     """
#     Config()는 singleton으로 동작함.
#     처음 config를 설정할 때, Config(config_name) 으로 설정을 해주면
#     후에는 Config()로 불러들여 사용하면 된다.

#     * tool의 main()에서 한번 Config(config_name)으로 설정하고,
#       다른 곳에서 Config를 참조할때, Config().db_root() 와 같은 식으로 사용해야 함.
#     """

#     def __init__(self, config_name):
#         # 설정 이름.
#         # 후에, espnet dataset 이름으로 활용함
#         self.name = config_name

#         config_path = SCRIPT.parent / "config" / f"{self.name}.yaml"
#         assert config_path.exists(), f"설정 파일 {config_path}가 없습니다."

#         with open(config_path, "r") as config_f:
#             config = yaml.load(config_f, yaml.FullLoader)

#         # 설정 파일 key 확인
#         for key in ["DB_root", "audio", "stft", "conv", "block", "policy"]:
#             assert (
#                 key in config.keys()
#             ), f"설정 파일은 반드시 '{key}' 항목을 포함해야 합니다."

#         # db root 설정
#         self.db_root = Path(config["DB_root"])
#         assert self.db_root.exists(), f"DB root {self.db_root}가 존재하지 않습니다."

#         self.audio = config["audio"]
#         for key in ["sample_rate"]:
#             assert (
#                 key in self.audio.keys()
#             ), f"audio 항목은 반드시 {key} 항목을 포함해야 합니다."

#         # stft/conv/block => block labeling 관련 설정
#         self.stft = config["stft"]
#         for key in ["n_fft", "hop_len"]:
#             assert (
#                 key in self.stft.keys()
#             ), f"stft 항목은 반드시 {key} 항목을 포함해야 합니다."

#         self.conv = config["conv"]
#         for key in ["kernel", "stride"]:
#             assert (
#                 key in self.conv.keys()
#             ), f"conv 항목은 반드시 {key} 항목을 포함해야 합니다."

#         self.block = config["block"]
#         for key in [
#             "block_size",
#             "previous_size",
#             "current_size",
#             "lookahead_size",
#             "hop_size",
#         ]:
#             assert (
#                 key in self.block.keys()
#             ), f"block 항목은 반드시 {key} 항목을 포함해야 합니다."

#         # traning utterance 생성 관련 policy
#         self.policy = config["policy"]
#         for key in ["filter", "boundary", "block", "label"]:
#             assert (
#                 key in self.policy.keys()
#             ), f"policy 항목은 반드시 {key} 항목을 포함해야 합니다."

#         print(f"============ Configuration =============")
#         print(f"- name:               {self.name}")
#         print(f"- audio --------------------------------")
#         print(f"    * sample rate:    {self.audio['sample_rate']}")
#         print(f"- stft --------------------------------")
#         print(f"    * n fft:          {self.stft['n_fft']}")
#         print(f"    * hop len:        {self.stft['hop_len']}")
#         print(f"- conv --------------------------------")
#         print(f"    * kernel:         {self.conv['kernel']}")
#         print(f"    * stride:         {self.conv['stride']}")
#         print(f"- block --------------------------------")
#         print(f"    * block size:     {self.block['block_size']}")
#         print(f"    * previous size:  {self.block['previous_size']}")
#         print(f"    * current size:   {self.block['current_size']}")
#         print(f"    * lookahead size: {self.block['lookahead_size']}")
#         print(f"    * hop size:       {self.block['hop_size']}")
#         print(f"- policy --------------------------------")
#         print(f"    * filter:         {self.policy['filter']}")
#         print(f"    * boundary:       {self.policy['boundary']}")
#         print(f"    * block:          {self.policy['block']}")
#         print(f"    * label:          {self.policy['label']}")
#         print()
