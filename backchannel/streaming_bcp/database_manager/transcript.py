from typing import Optional
import pandas as pd
from config import Config


class Transcript:
    """
    주어진 transcript dataframe 읽어서, training utterance를 만들기 위한 다양한 기능을 제공함.
    또한, 시간 단위로 되어 있는 time stamp를 모두 sample index 형식의 time stamp로 전환함
    """

    def __init__(
        self,
        label: pd.DataFrame,
        word=None,
        boundary=None,
    ):
        sample_rate = Config.audio["sample_rate"]

        # norm.xlsx DataFrame
        self.label = label
        # time stamp => index 변환
        self.label["start"] = self.label["start"].apply(
            lambda t: int(round(t * sample_rate))
        )
        self.label["end"] = self.label["end"].apply(
            lambda t: int(round(t * sample_rate))
        )

        self._speakers = None
        self._frontchannels = None
        self._backchannels = None

    def speakers(self) -> list:
        """
        주어진 label의 speaker 목록을 반환
        """
        if self._speakers is None:
            self._speakers = list(sorted(self.label["speaker"].unique()))
        return self._speakers

    def frontchannels(self, speaker=None) -> pd.DataFrame:
        """
        front channel utterance들을 반환
        speaker가 주어지면, speaker에 해당하는 것만 반환
        """
        if self._frontchannels is None:
            self._frontchannels = self.label[self.label["backchannel"] == "NOBC"]

        if speaker:
            assert speaker in self.speakers(), f"알 수 없는 발화자: {speaker}"
            return self._frontchannels[self._frontchannels["speaker"] == speaker]

        return self._frontchannels

    def backchannels(self, speaker=None) -> pd.DataFrame:
        """
        back channel utterance들을 반환
        speaker가 주어지면, speaker에 해당하는 것만 반환
        """
        if self._backchannels is None:
            self._backchannels = self.label[self.label["backchannel"] != "NOBC"]

        if speaker:
            assert speaker in self.speakers(), f"알 수 없는 발화자: {speaker}"
            return self._backchannels[self._backchannels["speaker"] == speaker]

        return self._backchannels

    def words(self, utt: pd.Series) -> Optional[pd.DataFrame]:
        """
        주어진 frontchannel utt에 해당하는 words alignment data frame 반환
        없거나 길이가 0이면 None
        """
        assert utt.backchannel == "NOBC", f"주어진 발화는 backchannel 발화입니다. {utt}"

        if self.word is None:
            return None

        words = self.word[
            (self.word["speaker"] == utt.speaker)  # 동일 화자
            & (self.word["segment_idx"] == utt.segment_idx)  # 동일 세그먼트
        ]
        if len(words) > 0:
            return words
        else:
            return None

    def following(self, utt: pd.Series) -> Optional[pd.Series]:
        """
        주어진 frontchannel utt에 뒤따르는 frontchannel utterance 반환
        없다면 None
        """
        assert utt.backchannel == "NOBC", f"주어진 발화는 backchannel 발화입니다. {utt}"

        # 주어진 발화가 시작된 뒤에 시작된 frontchannel 발화 중, 가장 빨리 발생한 것.
        # 시간 순으로 배열되어 있으므로, 그냥 [0] 번째 것 반환
        follow_utt = self.frontchannels()[self._frontchannels["start"] > utt.start]
        if len(follow_utt) > 0:
            return follow_utt.iloc[0]
        else:
            return None

    def bc_responses(self, utt: pd.Series) -> Optional[pd.DataFrame]:
        """
        주어진 frontchannel utt에 대한 반응 BC utterance들 반환
        """
        assert utt.backchannel == "NOBC", f"주어진 발화는 backchannel 발화입니다. {utt}"

        # 전체 bc
        bcs = self.backchannels()
        # 주어진 발화자가 아닌 상대방이 반응한 bc만 수집
        # 조건 0. bc 반응이므로, 내가 아닌 상대가 발화했을 것
        bcs = bcs[bcs["speaker"] != utt.speaker]

        # 주어진 frontchannel utt에 대한 반응 BC 수집
        # 조건 1. utt가 시작된 후에 발생했을 것
        bc_responses = bcs[bcs["start"] > utt.start]

        # 주어진 utt에 이어지는 front utt가 있다면
        # 조건 2. following utt 시작 전에 발생했을 것
        following = self.following(utt)
        if following is not None:
            bc_responses = bc_responses[bc_responses["start"] < following.start]

        # 1개 이상 있으면 pd.DataFrame
        if len(bc_responses) > 0:
            return bc_responses
        # 없으면 None
        else:
            return None
