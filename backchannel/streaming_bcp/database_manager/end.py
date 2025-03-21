"""
training utterance를 만드는 과정에서,
training utterance의 boundary를 결정하는 policy
"""

from dataclasses import dataclass
from typing import List
import pandas as pd


""" #########################################################
training utterance boundary policy abstract class
"""  #########################################################


class EndPolicy:
    def set_end(
        self, frontchannel: pd.Series, backchannels: pd.DataFrame, following: pd.Series
    ) -> tuple:
        """
        training.Utterance 인스턴스를 받아서,
        다양한 요소들을 활용해 training utterance의
        시작, 종료 시점을 반환
        """
        raise NotImplementedError()


class Default(EndPolicy):
    """
    frontchannel utterance의 시작, 종료를 그대로 반환
    """

    def set_end(
        self, frontchannel: pd.Series, backchannels: pd.DataFrame, following: pd.Series
    ) -> tuple:
        # 주어진 front-channel 발화의 시작과 종료를 그대로 활용
        return frontchannel.start, frontchannel.end


class Max(EndPolicy):
    """
    다음 utterance의 시작점 까지 최대한 길게 가져감
    """

    def set_end(
        self, frontchannel: pd.Series, backchannels: pd.DataFrame, following: pd.Series
    ) -> tuple:

        # dialogue에서 마지막 발화
        if following is None:

            # 백채널도 없으면
            if backchannels is None:
                return frontchannel.start, frontchannel.end

            # 마지막 백채널의 종료 지점까지 포함하도록
            else:
                # 마지막 bc 종료 지점
                last_bc = backchannels["end"].max()
                # 발화 종료와 마지막 bc 종료 지점 중 더 먼 것을 선택
                return frontchannel.start, max(frontchannel.end, last_bc)

        # following이 있음.
        else:

            # following과 overlap
            if frontchannel.end > following.start:
                # following과 overlap이 생기는 경우는
                # 상대가 내 발화도중 끼어든 상황 뿐임.
                # 상대가 끼어든 것이기 때문에 bc following.start 보다 늦게 발생할 수없음.
                # 이런 경우는 온전히 fronchannel 발화 시간을 유지
                return frontchannel.start, frontchannel.end
            # overlap 없는 경우
            else:
                # following 시작 전까지
                # following은 내가 이어서 말하거나,
                # 상대가 받아서 말하는 경우 발생
                # 이 때, bc가 following 시작 후 나타나는 경우는 내가 이어서 말하는 경우
                # 이 때는, 해당 bc는 다음 발화에 대한 bc가 되기 때문에
                # bc가 있건 없건 상관 없이 following의 start를 종료 지점으로 설정
                return frontchannel.start, following.start


class MaxBC(EndPolicy):
    """
    기본적으로 Default policy를 따르되,
    front-channel 발화가 끝난 후에 BC 반응이 있다면,
    그 BC 반응이 포함될 수 있도록, BC의 종료시까지 포함 시킴.
    다만, following이 있을 때, BC 반응이 Following의 시작을 넘어서까지 종료되지 않을 수 있음.
    이런 경우, Following의 시작까지 포함 시킴.
    """

    def set_end(
        self, frontchannel: pd.Series, backchannels: pd.DataFrame, following: pd.Series
    ) -> tuple:
        end = frontchannel.end

        # 발화가 종료된 후, 뒤따라 발생한 BC에 대한 labeling을 하기 위해
        # 발화 종료 시점을 BC의 종료 시점까지 늘림
        if backchannels is not None:
            # BC response 중, 종료 지점이 가장 긴 것을 선택
            last_bc = backchannels["end"].max()
            end = max(end, last_bc)

            # 만약 다음 frontchannel 발화가 있으면,
            if following is not None:
                # 다음 발화의 시작 지점과 비교해서 더 짧은 것을 종료 지점으로 잡음.
                end = min(end, following.start)

        return frontchannel.start, end


class MaxOnlyBC(EndPolicy):
    """
    front-channel 발화가 끝난 후에 BC 반응이 있다면,
    다음 utterance의 시작점 까지 최대한 길게 가져감
    """

    def set_end(
        self, frontchannel: pd.Series, backchannels: pd.DataFrame, following: pd.Series
    ) -> tuple:

        # dialogue에서 마지막 발화
        if following is None:

            # 백채널도 없으면
            if backchannels is None:
                return frontchannel.start, frontchannel.end

            # 마지막 백채널의 종료 지점까지 포함하도록
            else:
                # 마지막 bc 종료 지점
                last_bc = backchannels["end"].max()
                # 발화 종료와 마지막 bc 종료 지점 중 더 먼 것을 선택
                return frontchannel.start, max(frontchannel.end, last_bc)

        # following이 있음.
        else:

            # following과 overlap
            if frontchannel.end > following.start:
                # following과 overlap이 생기는 경우는
                # 상대가 내 발화도중 끼어든 상황 뿐임.
                # 상대가 끼어든 것이기 때문에 bc following.start 보다 늦게 발생할 수없음.
                # 이런 경우는 온전히 fronchannel 발화 시간을 유지

                return frontchannel.start, frontchannel.end
            # overlap 없는 경우
            else:
                # following 시작 전까지
                # following은 내가 이어서 말하거나,
                # 상대가 받아서 말하는 경우 발생
                # 이 때, bc가 following 시작 후 나타나는 경우는 내가 이어서 말하는 경우
                # 이 때, 해당 bc는 다음 발화에 대한 bc가 되기 때문에
                # bc가 있는 것에 대해서만 following의 start를 종료 지점으로 설정

                # 백채널이 있고 백채널 마지막 시작이 following 시작보다 빠를 경우에 following 시작을 종료로 설정
                if (
                    backchannels is not None
                    and frontchannel.end <= backchannels["start"].max()
                    and backchannels["start"].max() < following.start
                ):
                    return frontchannel.start, following.start
                # 백채널이 없는 경우
                else:
                    return frontchannel.start, frontchannel.end


class End:
    """
    정책을 이름을 바탕으로 class를 반환해줌.
    """

    class_map = {
        "Default": Default,
        "Max": Max,
        "MaxBC": MaxBC,
        "MaxOnlyBC": MaxOnlyBC,
    }

    @classmethod
    def get(cls, name, *args, **kwargs) -> EndPolicy:
        return cls.class_map[name](*args, **kwargs)
