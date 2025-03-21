"""
작성한 espent dataset을 탐색해서,
데이터에 관한 통계 작성
"""

import argparse
from dataclasses import dataclass

from config import Config
from locator import Resource
import numpy as np


class Stat:

    bc_category_map = {
        "NOBC": "NOBC",
        "CONTINUER": "CONTINUER",
        "UNDERSTANDING": "UNDERSTANDING",
        "EMPATHETIC": "EMPATHETIC",
        "AFFIRMATIVE": "EMPATHETIC",
        "ASSESSMENT": "EMPATHETIC",
        "REQUEST_CONFIRMATION": "EMPATHETIC",
        "REQUEST_COMFIRMATION": "EMPATHETIC",  # 데이터 베이스에 오타있음
        "NEGATIVE_SURPRISE": "EMPATHETIC",
        "POSITIVE_SURPRISE": "EMPATHETIC",
        "PAUSE": "PAUSE",
        "BOUNDARY": "BOUNDARY",
    }

    def __init__(self, name) -> None:
        self.name = name
        self.block_labels = list()

    def append_labels(self, labels: list):
        self.block_labels.append(
            list(map(lambda label: self.bc_category_map[label], labels))
        )

    def n_utterances(self):
        return len(self.block_labels)

    def n_blocks(self):
        return sum([len(labels) for labels in self.block_labels])

    def n(self, type):
        """[CONTINUER, UNDERSTANDING, EMPATHETIC, PAUSE, BOUNDARY]"""
        return np.sum(
            [
                1 if label == type else 0
                for labels in self.block_labels
                for label in labels
            ]
        )

    def n_bc(self):
        return np.sum(
            [
                1 if label != "NOBC" else 0
                for labels in self.block_labels
                for label in labels
            ]
        )

    def n_nobc(self):
        return np.sum(
            [
                1 if label == "NOBC" else 0
                for labels in self.block_labels
                for label in labels
            ]
        )

    def __repr__(self):
        text = ""
        text += f"============== {self.name} =============\n"
        text += f"--- Utterances: {self.n_utterances()}\n"
        text += f"------- Blocks: {self.n_blocks()}\n"
        text += f"--------- NOBC: {self.n_nobc()} ({self.n_nobc() / self.n_blocks():0.3f})\n"
        text += (
            f"------------BC: {self.n_bc()} ({self.n_bc() / self.n_blocks():0.3f})\n"
        )
        text += f"-    CONTINUER: {self.n('CONTINUER')} ({self.n('CONTINUER') / self.n_blocks():0.3f})\n"
        text += f"-UNDERSTANDING: {self.n('UNDERSTANDING')} ({self.n('UNDERSTANDING') / self.n_blocks():0.3f})\n"
        text += f"-   EMPATHETIC: {self.n('EMPATHETIC')} ({self.n('EMPATHETIC') / self.n_blocks():0.3f})\n"
        text += f"-        PAUSE: {self.n('PAUSE')} ({self.n('PAUSE')/self.n_blocks():0.3f})\n"
        text += f"-     BOUNDARY: {self.n('BOUNDARY')} ({self.n('BOUNDARY')/self.n_blocks():0.3f})\n"
        return text


class SWBDStat:

    bc_category_map = {
        "NOBC": "NOBC",
        "CONTINUER": "CONTINUER",
        "ASSESSMENT": "ASSESSMENT",
    }

    def __init__(self, name) -> None:
        self.name = name
        self.block_labels = list()

    def append_labels(self, labels: list):
        self.block_labels.append(
            list(map(lambda label: self.bc_category_map[label], labels))
        )

    def n_utterances(self):
        return len(self.block_labels)

    def n_blocks(self):
        return sum([len(labels) for labels in self.block_labels])

    def n(self, type):
        """[CONTINUER, ASSESSMENT]"""
        return np.sum(
            [
                1 if label == type else 0
                for labels in self.block_labels
                for label in labels
            ]
        )

    def n_bc(self):
        return np.sum(
            [
                1 if label != "NOBC" else 0
                for labels in self.block_labels
                for label in labels
            ]
        )

    def n_nobc(self):
        return np.sum(
            [
                1 if label == "NOBC" else 0
                for labels in self.block_labels
                for label in labels
            ]
        )

    def __repr__(self):
        text = ""
        text += f"============== {self.name} =============\n"
        text += f"--- Utterances: {self.n_utterances()}\n"
        text += f"------- Blocks: {self.n_blocks()}\n"
        text += (
            f"-----Max Block: {np.max([len(labels) for labels in self.block_labels])}\n"
        )
        text += (
            f"-----Min Block: {np.min([len(labels) for labels in self.block_labels])}\n"
        )
        text += f"----Mean Block: {np.mean([len(labels) for labels in self.block_labels])}\n"
        text += f"--------- NOBC: {self.n_nobc()} ({self.n_nobc() / self.n_blocks():0.3f})\n"
        text += (
            f"------------BC: {self.n_bc()} ({self.n_bc() / self.n_blocks():0.3f})\n"
        )
        text += f"-    CONTINUER: {self.n('CONTINUER')} ({self.n('CONTINUER') / self.n_blocks():0.3f})\n"
        text += f"-   ASSESSMENT: {self.n('ASSESSMENT')} ({self.n('ASSESSMENT') / self.n_blocks():0.3f})\n"
        return text


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, default="bcko")
    parser.add_argument("name", type=str, default="base")
    args = parser.parse_args()

    # config 이름을 읽어와서, 설정. Config는 singleton 이므로, 아무곳에서나 생성해서 사용 가능함.
    Config.load(args.name, verbose=True)

    if args.dataset == "bcko":
        stats = {
            "Counseling.SelectStar.2022": Stat(name="Counseling.SelectStar.2022"),
            "Counseling.SelectStar.2023": Stat(name="Counseling.SelectStar.2023"),
            "Counseling.SMT.2023": Stat(name="Counseling.SMT.2023"),
            "train": Stat(name="train"),
            "valid": Stat(name="valid"),
            "test": Stat(name="test"),
        }

        for split in ["train", "valid", "test"]:
            label_path = Resource.espnet_label_path(split)
            with open(label_path, "r") as f:
                lines = f.readlines()

                for line in lines:
                    uid = line.split()[0]
                    name = ".".join(uid.split(".")[0:3])
                    labels = line.split()[1:]

                    stats[name].append_labels(labels)
                    stats[split].append_labels(labels)

    elif args.dataset == "swbd":
        stats = {
            "SWBD": SWBDStat(name="SWBD"),
            "train": SWBDStat(name="train"),
            "valid": SWBDStat(name="valid"),
            "test": SWBDStat(name="test"),
        }

        for split in ["train", "valid", "test"]:
            label_path = Resource.espnet_label_path(split)
            with open(label_path, "r") as f:
                lines = f.readlines()

                for line in lines:
                    labels = line.split()[1:]

                    stats["SWBD"].append_labels(labels)
                    stats[split].append_labels(labels)

    else:
        raise Exception("Invalid dataset name")

    for name, stat in stats.items():
        print(stat)

        # blocks = list()
        # n_blocks = 0
        # nobcs = 0
        # bcs = list()
        # n_bcs = 0

        # for line in lines:
        #     labels = line.split()[1:]
        #     blocks.append(len(labels))
        #     n_blocks += len(labels)
        #     nobcs += len([label for label in labels if label == "NOBC"])
        #     n_bcs += len([label for label in labels if label != "NOBC"])
        #     bcs.append(len([label for label in labels if label != "NOBC"]))

        # print(f"============== {split} =============")
        # print(f"--- Utterances: {len(lines)}")
        # print(f"------- Blocks: {n_blocks}")
        # print(f"-         mean: {np.mean(blocks)}")
        # print(f"-       median: {np.median(blocks)}")
        # print(f"-          max: {np.max(blocks)}")
        # print(f"-          min: {np.min(blocks)}")
        # print(f"--------- NOBC: {nobcs} ({nobcs/n_blocks:0.2f})")
        # print(f"------------BC: {n_bcs} ({n_bcs/n_blocks:0.2f})")
        # print(f"-         mean: {np.mean(bcs)}")
        # print(f"-       median: {np.median(bcs)}")
        # print(f"-          max: {np.max(bcs)}")
        # print(f"-          min: {np.min(bcs)}")
