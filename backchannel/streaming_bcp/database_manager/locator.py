"""
loader
"""

from typing import Optional
import pandas as pd
from pathlib import Path
import logging
import wave
import yaml
from config import Config


class Resource:
    """
    return Path of given resource
    """

    @classmethod
    def db_root(cls):
        return Config.db_root

    @classmethod
    def name(cls, db, vendor=None, publish=None, number_idx=-1):
        """
        주어진 내용을 바탕으로 이름을 반환
        """
        filename = f"{db}"

        if vendor is not None:
            filename += f".{vendor}"

        if publish is not None:
            filename += f".{publish}"

        if number_idx != -1:
            filename += f"-{number_idx:05d}"

        return filename

    @classmethod
    def get_uid(cls, name, dialogue, speaker, segment_idx):
        """
        주어진 인자를 바탕으로 uid 생성해줌
        """
        return f"{name}.{dialogue}.{speaker}.{segment_idx:04}"

    @classmethod
    def parse_uid(cls, uid):
        """
        주어진 uid를 바탕으로 인자들을 리턴
        """
        if len(uid.split(".")) == 4:
            return (
                uid.split(".")[0],
                uid.split(".")[1],
                uid.split(".")[2],
                int(uid.split(".")[3]),
            )
        else:
            return (
                ".".join(uid.split(".")[0:3]),
                uid.split(".")[-3],
                uid.split(".")[-2],
                int(uid.split(".")[-1]),
            )

    @classmethod
    def norm_path(cls, name: str) -> Path:
        """
        .norm 파일의 위치를 반환
        """
        for candidate_path in (cls.db_root() / "label" / "norm").iterdir():
            if candidate_path.name.startswith(name):
                path = candidate_path

        assert path is not None, f"{name}.norm 이 없습니다."

        return path

    @classmethod
    def word_path(cls, name: str) -> Path:
        """
        .word 파일의 위치를 반환
        """
        for candidate_path in (cls.db_root() / "label" / "word").iterdir():
            if candidate_path.name.startswith(name):
                path = candidate_path

        assert path is not None, f"{name}.word 이 없습니다."

        return path

    @classmethod
    def dialogues(cls, name) -> list:
        """
        주어진 db의 대화 목록을 반환
        """
        return pd.ExcelFile(cls.norm_path(name)).sheet_names

    @classmethod
    def norm(cls, name, dialogue) -> pd.DataFrame:
        return pd.ExcelFile(cls.norm_path(name)).parse(dialogue)

    @classmethod
    def word(cls, name, dialogue) -> pd.DataFrame:
        """
        .norm 파일 중, 주어진 대화의 내용을 pd.DataFrame으로 반환
        word alignment 파일은 sheet name이 {dialogue}.{speaker} 로 분리되어 있음.
        따라서, 두개를 모두 읽어서 합쳐서 줘야함.
        """
        xlsx = pd.ExcelFile(cls.word_path(name))
        sheets = xlsx.sheet_names

        dfs = list()
        for sheet in sheets:
            # speaker를 제외한 나머지
            if dialogue == ".".join(sheet.split(".")[:-1]):
                dfs.append(xlsx.parse(sheet))

        # 합쳐서 return
        return pd.concat(dfs)

    @classmethod
    def wav_path(cls, name, dialogue, speaker) -> Path:
        """
        주어진 항목에 따라 wave 파일 경로 반환
        """
        return cls.db_root() / "audio" / name / f"{dialogue}.speaker={speaker}.wav"

    @classmethod
    def dialogue_split_path(cls, name) -> Path:
        """
        split 정보가 담긴 yaml 파일의 경로를 반환
        """
        return cls.db_root() / "split" / f"{name}.split.dialogue.yaml"

    @classmethod
    def dialogue_split(cls, name) -> dict:
        """
        split 정보가 담긴 yaml 파일을 읽어서 dict로 반환
        """
        split_path = cls.dialogue_split_path(name)
        assert split_path.exists(), f"분할 파일 {split_path}이 존재하지 않습니다."

        with open(split_path, "r", encoding="utf-8") as split_f:
            split_dict = yaml.load(split_f, yaml.FullLoader)
        return split_dict

    @classmethod
    def utterance_split_path(cls, name) -> Path:
        return cls.db_root() / "split" / f"{name}.split.utterance.yaml"

    @classmethod
    def utterance_split(cls, name):
        split_path = cls.utterance_split_path(name)
        assert split_path.exists(), f"분할 파일 {split_path}이 존재하지 않습니다."

        with open(split_path, "r", encoding="utf-8") as split_f:
            split_dict = yaml.load(split_f, yaml.FullLoader)
        return split_dict

    @classmethod
    def espnet_dir(cls):
        return cls.db_root() / "espnet" / Config.name

    @classmethod
    def espnet_audio_path(cls, uid):
        return cls.espnet_dir() / "audios" / f"{uid}.wav"

    @classmethod
    def espnet_text_path(cls, split):
        return cls.espnet_dir() / "splits" / f"{split}.text"

    @classmethod
    def espnet_label_path(cls, split):
        return cls.espnet_dir() / "splits" / f"{split}.label"

    @classmethod
    def clean(cls):
        """
        espnet 안의 데이터 삭제
        """

        # 삭제
        cls.rm_dir(cls.espnet_dir(), verbose=True)

        # 생성
        cls.if_not_exist_mkdir(cls.espnet_dir() / "audios", verbose=True)
        cls.if_not_exist_mkdir(cls.espnet_dir() / "splits", verbose=True)

    @classmethod
    def rm_dir(cls, dir: Path, verbose=False):
        """
        디렉토리 안의 데이터 재귀적으로 삭제
        디렉토리 포함
        """
        if not dir.exists():
            if verbose:
                print(f"{str(dir)}가 존재하지 않습니다.")
            return

        if verbose:
            print(f"{str(dir)} 삭제.")
        if dir.is_dir():
            # 디렉토리인 경우, 하위 디렉토리 및 파일 탐색
            for child in dir.iterdir():
                cls.rm_dir(child)
            dir.rmdir()
        else:
            # 파일인 경우 삭제
            dir.unlink()

    @classmethod
    def if_not_exist_mkdir(cls, dir: Path, verbose=False) -> Path:
        """
        주어진 dpath에 directory가 있는 지 확인, 없으면 새로 생성함.
        """
        if not dir.is_dir():
            if verbose:
                print(f"{str(dir)} 경로에 디렉토리가 없으므로 새로 생성.")
            dir.mkdir(parents=True)
        else:
            if verbose:
                print(f"{str(dir)} 디렉토리 확인.")

        return dir
