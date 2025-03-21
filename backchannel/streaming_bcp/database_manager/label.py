"""
training utterance를 만드는 과정에서,
block labeling 방법에 관한 policy

"""

from dataclasses import dataclass
from typing import List
import pandas as pd

from config import Config


""" #########################################################
block labeling policy abstract class
"""  #########################################################


class AbsLabelPolicy:
    """
    미리 구한 blocks 는 오디오 파일을 기준, 인덱스 작성되어 있으므로,
    시작 지점을 고려한 offset 처리할 필요 없음.
    """

    def init_labels(self, n_block: int, n_event: int) -> List[List[str]]:
        ls = list()
        for _ in range(n_event):
            ls.append(["NOBC"] * n_block)

        return ls

    def noblock_labels(self, events) -> List[List[str]]:
        """
        신호가 너무 짧은 경우, block이 생성되지 않는 경우가 있음.
        """
        labels = self.init_labels(1, len(events))

        # event가 있는 경우, 그냥 덮어씀
        for i, timings in enumerate(events):
            for event in timings:
                labels[i][0] = event.category

        return labels

    def labeling(self, blocks, events) -> List[List[str]]:
        raise NotImplementedError()


class CurrentClassification(AbsLabelPolicy):
    """
    1. 현재 block의 current context에 BC 시작 시, 현재 block을 BC로 레이블링
    """

    def labeling(self, blocks, events) -> List[str]:

        # 너무 짧은 경우, block이 생성되지 않는 경우가 있음.
        if len(blocks) == 0:
            return self.noblock_labels(events)

        # label 초기화
        labels = self.init_labels(len(blocks), len(events))

        # label 종류 별 순회
        for i, timings in enumerate(events):

            for j, block in enumerate(blocks):
                start, end = block.current

                for event in timings:
                    # 주어진 event의 시작이, 해당 블럭의 current context 안에 있다면,
                    if event.start > start and event.start < end:
                        labels[i][j] = event.category

        return labels


class CurrentClassificationwithFilter(AbsLabelPolicy):
    """
    1. 현재 block의 current context에 BC 시작 시, 현재 block을 BC로 레이블링
    2. 하나의 event는 하나의 블록에만 할당될 수 있음
       즉, block.nxt.start와 block.current.end 사이에 발생한 event의 경우는 2개의 블록에 BC로 레이블링이 될 수 있음
    3. 첫번째 블록에서 BC가 발생 시에는 많은 정보가 없으므로 제외한다.
    """

    def labeling(self, blocks, events) -> List[str]:

        # 너무 짧은 경우, block이 생성되지 않는 경우가 있음.
        if len(blocks) == 0:
            return self.noblock_labels(events)

        # label 초기화
        labels = self.init_labels(len(blocks), len(events))

        # label 종류 별 순회
        for i, timings in enumerate(events):
            chosen = 0
            for j, block in enumerate(blocks[1:], start=1):
                start, end = block.current

                for event in timings[chosen:]:
                    # 주어진 event의 시작이, 해당 블럭의 current context 안에 있다면,
                    if event.start > start and event.start < end:
                        labels[i][j] = event.category
                        chosen += 1
                        break

        return labels


class Label:
    """
    정책을 이름을 바탕으로 class를 반환해줌.
    """

    class_map = {
        "CurrentClassification": CurrentClassification,
        "CurrentClassificationwithFilter": CurrentClassificationwithFilter,
    }

    @classmethod
    def get(cls, name, *args, **kwargs) -> AbsLabelPolicy:
        return cls.class_map[name](*args, **kwargs)


"""
###################### Timing ############################
"""


@dataclass
class EventTiming:
    start: int
    end: int
    category: str


class AbsEventPolicy:
    """
    주어진 정보를 활용해서 Timing 반환 [EventTiming(start, end, category)]
    """

    def timing(
        self, backchannels: pd.DataFrame, words: pd.DataFrame, boundary: str
    ) -> List[EventTiming]:
        raise NotImplementedError()


class BC(AbsEventPolicy):
    """
    BC 정보를 활용, timing 정보로 반환
    """

    def timing(
        self, backchannels: pd.DataFrame, words: pd.DataFrame, boundary: str
    ) -> List[EventTiming]:
        if backchannels is None:
            return []
        return [
            EventTiming(start=bc.start, end=bc.end, category=bc.backchannel)
            for _, bc in backchannels.iterrows()
        ]


class BCwFilter(AbsEventPolicy):
    """
    BC 정보를 활용, timing 정보로 반환
    """

    def timing(
        self, backchannels: pd.DataFrame, words: pd.DataFrame, boundary: str
    ) -> List[EventTiming]:
        if backchannels is None:
            return []
        old_events = [
            EventTiming(start=bc.start, end=bc.end, category=bc.backchannel)
            for _, bc in backchannels.iterrows()
        ]

        # 이전 백채널과 현재 백채널의 start 차이가 1sec이상인 것들만 살림
        new_events = [old_events[0]]
        for i in range(1, len(old_events)):
            if (
                old_events[i].start - old_events[i - 1].start
                > Config.audio["sample_rate"]
            ):
                new_events.append(old_events[i])
        return new_events


class Event:
    """
    정책을 이름을 바탕으로 class를 반환해줌.
    """

    class_map = {
        "BC": BC,
        "BCwFilter": BCwFilter,
    }

    @classmethod
    def get(cls, name, *args, **kwargs) -> AbsEventPolicy:
        return cls.class_map[name](*args, **kwargs)
