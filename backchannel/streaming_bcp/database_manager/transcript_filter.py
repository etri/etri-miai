"""
training utterance를 만드는 과정에서,
filtering

dataframe을 받아서 갈무리 해서 반환함.
"""

import pandas as pd


""" #########################################################
block policy abstract class
"""  #########################################################


class AbsFilterPolicy:
    def filter(self, label: pd.DataFrame, verbose=False) -> pd.DataFrame:
        raise NotImplementedError()


class PassThrough(AbsFilterPolicy):
    """
    1. 따로 필터링하지 않음.
    """

    def filter(self, label: pd.DataFrame, verbose=False) -> pd.DataFrame:
        if verbose:
            print(f"PassThrough filtered: {len(label)} -> {len(label)}")
        return label


class Empty(AbsFilterPolicy):
    """
    2. NaN or empty string filter
    """

    def filter(self, label: pd.DataFrame, verbose=False) -> pd.DataFrame:
        filtered = label[~((label["norm_text"].isna()) | (label["norm_text"] == ""))]
        if verbose:
            print(f"Empty filtered: {len(label)} -> {len(filtered)}")
        return filtered


class TooShort(AbsFilterPolicy):
    """
    3. Too short utterance filtering
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def filter(self, label: pd.DataFrame, verbose=False) -> pd.DataFrame:

        # front-channel utterance 중에, 길이가 threhold보다 작은 것
        drop_index = (label["backchannel"] == "NOBC") & (
            label["end"] - label["start"] < self.threshold
        )
        filtered = label[~drop_index]

        if verbose:
            print(f"TooShort(<1s) filtered: {len(label)} -> {len(filtered)}")
            print(label[drop_index]["norm_text"])
        return filtered


class Filter:
    """
    정책을 이름을 바탕으로 class를 반환해줌.
    """

    class_map = {"PassThrough": PassThrough, "Empty": Empty, "TooShort": TooShort}

    @classmethod
    def get(cls, name, *args, **kwargs) -> AbsFilterPolicy:
        return cls.class_map[name](*args, **kwargs)
