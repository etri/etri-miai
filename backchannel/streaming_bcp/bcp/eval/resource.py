"""
파일 읽기 및 쓰기를 위한 함수들을 정의한 파일입니다.
"""

import json
import yaml

from pathlib import Path
from typing import Dict


class Resource:
    """
    파일 읽기 및 쓰기를 위한 함수들을 정의한 클래스.
    """

    @staticmethod
    def load_file_for_label(label_path: Path) -> Dict[str, str]:
        """
        라벨 파일을 읽어서 문자열로 반환.
        예) 백채널 예측 파일 및 정답 파일
        Args:
            label_path (Path): 파일 경로

        Returns:
            dict[str, str]:
                key: utterance id
                value: block 별 라벨 정보
        """
        if not label_path.exists():
            raise FileNotFoundError(f"{label_path}이 존재하지 않습니다.")

        utts = dict()
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                key, *labels = line.split()
                utts.setdefault(key, labels)

        # sort by key
        utts = dict(sorted(utts.items()))
        return utts

    @staticmethod
    def load_config_file(config_path: Path) -> Dict:
        """
        config 파일을 읽어서 dict로 반환.
        Args:
            file_path (Path): 파일 경로

        Returns:
            dict: 모델 configure 정보
        """
        if not config_path.exists():
            raise FileNotFoundError(f"{config_path}이 존재하지 않습니다.")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    @staticmethod
    def save_result_file(result_path: Path, new_result: Dict):
        def convert_value_to_str(datas: Dict):
            new_datas = {}
            for key, value in datas.items():
                if isinstance(value, dict):
                    new_datas[key] = convert_value_to_str(value)
                elif isinstance(value, list):
                    new_datas[key] = str(value)
                elif isinstance(value, Path):
                    new_datas[key] = str(value)
                else:
                    new_datas[key] = value
            return new_datas

        """
        결과 파일을 json 형태로 저장.

        기존의 결과파일이 존재한다면 읽어와서 덮어쓰기를 수행.
        Args:
            result_path (Path): 파일 경로
            result (dict): 결과 값
        """

        new_result = convert_value_to_str(new_result)

        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
                result.update(new_result)
        else:
            result = new_result

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        print(f"Result is saved at {result_path}")
