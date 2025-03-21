from typing import Optional
import pandas as pd
from tqdm import tqdm

from config import Config
from locator import Resource
from blocks import BlockBound
from end import End
from label import Label, Event


class Utterance:
    """
    Training 단위가 되는 utterance
    frontchannel utterance
    backchannel utterances로 구성
    """

    def __init__(
        self,
        name,
        dialogue,
        frontchannel: pd.Series,
        following: Optional[pd.Series] = None,
        backchannels: Optional[pd.DataFrame] = None,
        words: Optional[pd.DataFrame] = None,
        boundary: Optional[str] = None,
    ):
        self.name = name
        self.dialogue = dialogue
        self.frontchannel = frontchannel
        self.following = following  # nullable
        self.backchannels = backchannels  # nullable
        self.words = words  # nullable
        self.boundary = boundary  # nullable

        self.speaker = self.frontchannel.speaker

        # 정책에 따라서 후처리, tuple(start, end) in sample index
        # 예를 들어, frontchannel utterance가 끝나고 반응한 BC가 있으면, 그 시간을 고려해서, training utterance의 종료 시간을 조정하는 정책을 반영
        self.start, self.end = self._set_end()

        # block 경계
        self._blocks = self._set_blocks()

        # block에 따른 labeling
        self._labels = self._labeling()  # list 형태로 block labeling

    @property
    def uid(self):
        return Resource.get_uid(
            self.name,
            self.dialogue,
            self.frontchannel.speaker,
            self.frontchannel.segment_idx,
        )

    @property
    def text(self):
        return self.frontchannel.norm_text

    @property
    def labels(self):
        return self._labels

    @property
    def blocks(self):
        return self._blocks

    def _set_end(self):
        """
        Training utterance의 시작 / 종료 지점 설정
        """
        return End.get(Config.policy["end"]).set_end(
            self.frontchannel, self.backchannels, self.following
        )

    def _set_blocks(self):
        """
        결정된 start, end에 따라서 block 경계 계산
        """
        # block 경계 계산기
        return BlockBound(
            **Config.block,
            stft_kwargs=Config.stft,
            conv_kwargs=Config.conv,
        ).blocks(self.start, self.end)

    def _labeling(self):
        """
        정책에 따라 생성한 label list를 반환
        """

        # label type 별로, timing 정보를 획득
        """
        timing=[
            [EventTiming, ...,],  # BC 관련 timings
            [EventTiming, ...,],  # 문장/절 경계 timings
            [EventTiming, ...,],  # Pause timings
        ]
        각 EventTiming = {start: start index, end: end index, category: 종류}
        """
        events = [
            Event.get(**label_type).timing(self.backchannels, self.words, self.boundary)
            for label_type in Config.policy["label"]["type"]
        ]
        # labeling 시에, 중요도가 높은 것을 나중에 labeling 하여 덮어씌우는 전략으로 중요도를 반영함.
        # 따라서, 덜 중요한 것 먼저 label을 작성하기 위해 reverse

        """
        labels=[
            ["NOBC", "CONTINUER", "NOBC", ...], # 첫번째 type labels
            ["NOBC", "NOBC", "BOUNDARY", ...],  # 두번째 type labels
            ["NOBC", "PAUSE", "NOBC", ...],     # 세번째 type labels
        ]
        """
        labels = Label.get(Config.policy["label"]["annotation"]).labeling(
            self.blocks, events
        )

        # type 별 labels 들을 합쳐서 하나의 label로
        # And 인 경우
        if Config.policy["label"]["join"] == "And":
            integrated_label = list()
            n_type = len(labels)  # label 종류 갯수
            n_block = len(labels[0])

            # j 번째 block에 대해
            for j in range(n_block):
                # 모든 type 별 j 번째 블록이 모두 NOBC가 아니면,
                if all([labels[i][j] != "NOBC" for i in range(n_type)]):
                    # 가장 마지막 타입의 레이블 할당 (우선순위 고려 때문)
                    integrated_label.append(labels[0][j])
                else:
                    integrated_label.append("NOBC")

            return integrated_label

        # Or인 경우, 그냥 순서대로 덮어쓰면 됨. 다만, 우선순위가 높은것이 앞에 있으므로 뒤집어야함.
        elif Config.policy["label"]["join"] == "Or":
            n_block = len(labels[0])
            integrated_label = ["NOBC"] * n_block

            labels.reverse()  # 우선순위 때문에 뒤집음
            # 타입별
            for type_labels in labels:
                # 블럭 별
                for j, label in enumerate(type_labels):
                    if label != "NOBC":
                        integrated_label[j] = label

            return integrated_label

        else:
            raise Exception(f"Unknown label join: {Config.policy['label']['join']}")


class LabelWriter:

    def __init__(self) -> None:

        self.split_type = Config.policy["split"]
        self._split_info = dict()

    def split_info(self, name):
        if name not in self._split_info.keys():
            if self.split_type == "Dialogue":
                self._split_info[name] = Resource.dialogue_split(name)
            elif self.split_type == "Utterance":
                self._split_info[name] = Resource.utterance_split(name)
            else:
                raise Exception(f"Unknown split type: {self.split_type}")

        return self._split_info[name]

    def get_split(self, uid):

        name, dialogue, _, _ = Resource.parse_uid(uid)

        if self.split_type == "Dialogue":
            return self.split_info(name)[dialogue]
        elif self.split_type == "Utterance":
            return self.split_info(name)[uid]
        else:
            raise Exception(f"Unknown split type: {self.split_type}")

    # @classmethod
    # def get_split(cls, uid, split_type):

    #     name, dialogue, _, _ = Resource.parse_uid(uid)

    #     # split type에 따라 split 여부 결정
    #     if split_type == "Dialogue":
    #         # dialoge 파일은 {dialogue: train, ...}
    #         split_info = Resource.dialogue_split(name)
    #         split = split_info[dialogue]

    #     elif split_type == "Utterance":
    #         # utterance 파일은 {uid: train, ...}
    #         split_info = Resource.utterance_split(name)
    #         split = split_info[uid]

    #     else:
    #         raise Exception(f"Unknown split method: {split_type}")

    #     return split

    # @classmethod
    # def write_text(cls, utt: Utterance, split_type):
    #     text = utt.text
    #     uid = utt.uid

    #     # split type에 따라 split 여부 결정
    #     split = cls.get_split(uid, split_type)

    #     # exclude일 경우, 작성하지 않고 넘어감.
    #     if split == "exclude":
    #         return

    #     Resource.if_not_exist_mkdir(Resource.espnet_dir() / "splits")

    #     # 작성
    #     with open(Resource.espnet_text_path(split), "a") as af:
    #         af.write(f"{uid} {text}\n")

    # @classmethod
    # def write_labels(cls, utt: Utterance, split_type):
    #     labels = utt.labels
    #     uid = utt.uid

    #     # split type에 따라 split 여부 결정
    #     split = cls.get_split(uid, split_type)

    #     # exclude일 경우, 작성하지 않고 넘어감.
    #     if split == "exclude":
    #         return

    #     Resource.if_not_exist_mkdir(Resource.espnet_dir() / "splits")

    #     # 작성
    #     with open(Resource.espnet_label_path(split), "a") as af:
    #         af.write(f"{uid} {' '.join(labels)}\n")

    def write(self, results):
        """
        parallel processing의 결과로 나온 것들을 작성할 때,
        results = [
            (uid, text, labels)
        ]
        """
        Resource.if_not_exist_mkdir(Resource.espnet_dir() / "splits")

        # open files
        text_file = dict(
            train=open(Resource.espnet_text_path("train"), "w"),
            valid=open(Resource.espnet_text_path("valid"), "w"),
            test=open(Resource.espnet_text_path("test"), "w"),
        )

        label_file = dict(
            train=open(Resource.espnet_label_path("train"), "w"),
            valid=open(Resource.espnet_label_path("valid"), "w"),
            test=open(Resource.espnet_label_path("test"), "w"),
        )

        for uid, text, labels in (pbar := tqdm(results)):
            pbar.set_description(f"writing {uid}")

            # uid로 부터 split을 구함
            split = self.get_split(uid)
            if split == "exclude":
                continue

            text_file[split].write(f"{uid} {text}\n")
            label_file[split].write(f"{uid} {' '.join(labels)}\n")

        for _, file_ in text_file.items():
            file_.close()

        for _, file_ in label_file.items():
            file_.close()
